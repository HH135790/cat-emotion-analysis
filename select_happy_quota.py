# -*- coding: utf-8 -*-
"""happy 池配额选样 —— 修复 purr 偏置
====================================================================
问题: happy 若按 YAMNet 猫叫置信度取 top-150, 选进来的几乎全是 purr(呼噜),
而 V2 测试集的 happy 是短促喵叫 —— 模型学到的 "happy" 与 resting(呼噜)同区,
跨域时 resting 被成片判成 happy。
方案: 对 happy 全部候选打 Purr/Meow 子类分数, meow 主导 / purr 主导
各取 75 条 (不足互补), 组成 150 条。
输入: data/train/*.wav + data/labels_train.csv + data/mapping.csv
输出: data/happy_quota_selection.json (finetune_final_hybrid.py 自动读取)
"""
import os, sys, csv, json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[1]
YAMNET_PKG = ROOT / 'yamnet'
TRAIN_DIR = ROOT / 'data' / 'train'
LABELS_CSV = ROOT / 'data' / 'labels_train.csv'   # filename, v2_emotion, ...
CONF_CSV = ROOT / 'data' / 'mapping.csv'          # filename, confidence, ...
OUT_JSON = ROOT / 'data' / 'happy_quota_selection.json'
SR, FIXED = 16000, 16000 * 6
TOP_K = 150


def patch_keras3_allow_slash():
    import keras.src.ops.operation as _op_mod
    def _p(self, name=None, *a, **kw):
        if name is None:
            import keras.src.utils.naming as _nm
            name = _nm.auto_name(self.__class__.__name__)
        if not isinstance(name, str): raise ValueError('name must be str')
        self.name = name
        self._inbound_nodes = []; self._outbound_nodes = []
        try:
            from keras.src.backend.common import global_state
            self._parent_path = global_state.get_global_attribute(
                'name_scope_stack', default=[], set_to_default=True)[-1].name
        except Exception: self._parent_path = ""
        self._remat_mode = None; self._obj_reference = None
    _op_mod.Operation.__init__ = _p
    import keras.src.backend.common.name_scope as _ns_mod
    def _pn(self, name, caller=None, deduplicate=True, override_parent=None):
        if not isinstance(name, str): raise ValueError('name must be str')
        self.name = name; self.caller = caller; self.deduplicate = deduplicate
        self.override_parent = override_parent
        if override_parent is None and deduplicate and getattr(caller, '_parent_path', None) is not None:
            self.override_parent = caller._parent_path
        self._pop_on_exit = False
    _ns_mod.name_scope.__init__ = _pn


def load_wav(path):
    y, native = sf.read(str(path), dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    if native != SR:
        g = gcd(int(native), SR)
        y = resample_poly(y, SR // g, int(native) // g).astype('float32')
    if len(y) < FIXED:
        y = np.tile(y, FIXED // len(y) + 1)[:FIXED]
    return y[:FIXED]


def main():
    patch_keras3_allow_slash()
    sys.path.insert(0, str(YAMNET_PKG))
    import params as yamnet_params
    import yamnet as yamnet_model
    import features as features_lib
    from tensorflow import keras

    PARAMS = yamnet_params.Params()
    idx = {}
    with open(YAMNET_PKG / 'yamnet_class_map.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            idx[r['display_name']] = int(r['index'])
    PURR, MEOW = idx['Purr'], idx['Meow']

    def _pre(x):
        wp = features_lib.pad_waveform(x, PARAMS)
        _, patches = features_lib.waveform_to_log_mel_spectrogram_patches(wp, PARAMS)
        return patches

    inp = keras.Input(batch_shape=(None,), dtype='float32', name='wave')
    preds_t, _ = yamnet_model.yamnet(
        keras.layers.Lambda(_pre, output_shape=(None, PARAMS.patch_frames, PARAMS.patch_bands),
                            name='logmel_patches')(inp), PARAMS)
    scores_model = keras.Model(inp, preds_t)

    import h5py
    bb_map = {l.name: l for l in scores_model.layers if l.weights}
    slot_order = {'kernel': 0, 'bias': 1, 'beta': 0, 'moving_mean': 1, 'moving_variance': 2}
    by_layer = {}
    with h5py.File(str(YAMNET_PKG / 'yamnet.h5'), 'r') as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                p = name.rsplit('/', 1)[0]
                c = p.split('/')
                if len(c) % 2 == 0 and c[:len(c)//2] == c[len(c)//2:]:
                    by_layer.setdefault('/'.join(c[:len(c)//2]), []).append(
                        (slot_order.get(name.rsplit('/', 1)[1].replace(':0', ''), 9), obj[()]))
        f.visititems(_visit)
    n = 0
    for lname, arrs in by_layer.items():
        if lname in bb_map:
            arrs.sort(key=lambda x: x[0])
            ws = [np.asarray(a) for _, a in arrs]
            layer = bb_map[lname]
            if len(ws) == len(layer.weights) and all(w.shape == k.shape for w, k in zip(ws, layer.weights)):
                layer.set_weights(ws); n += 1
    print(f'权重迁移(h5直读): {n}/{len(bb_map)} 层', flush=True)
    assert n >= 50, '权重迁移异常'

    # happy 候选 (置信度非空)
    conf = {}
    with open(CONF_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try: conf[r['filename']] = float(r['confidence'])
            except (ValueError, KeyError): pass
    cands = []
    with open(LABELS_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['v2_emotion'] == 'happy' and r['filename'] in conf:
                p = TRAIN_DIR / r['filename']
                if p.exists():
                    cands.append(p)
    print(f'happy 候选: {len(cands)}', flush=True)

    meow_grp, purr_grp, rest = [], [], []
    for i, p in enumerate(cands, 1):
        try:
            pr = np.asarray(scores_model(load_wav(p)))
            cm, cp = float(pr[:, MEOW].max()), float(pr[:, PURR].max())
            d = {'path': str(p.relative_to(ROOT)).replace('\\', '/'),
                 'conf_meow': round(cm, 4), 'conf_purr': round(cp, 4)}
            if cm >= cp and cm >= 0.05: meow_grp.append(d)
            elif cp > cm and cp >= 0.05: purr_grp.append(d)
            else: rest.append(d)
        except Exception:
            rest.append({'path': str(p.relative_to(ROOT)).replace('\\', '/'),
                         'conf_meow': -1.0, 'conf_purr': -1.0})
        if i % 100 == 0:
            print(f'  已打分 {i}/{len(cands)} (meow {len(meow_grp)} / purr {len(purr_grp)})', flush=True)

    meow_grp.sort(key=lambda d: -d['conf_meow'])
    purr_grp.sort(key=lambda d: -d['conf_purr'])
    q = TOP_K // 2
    sel = meow_grp[:q] + purr_grp[:q]
    if len(sel) < TOP_K:
        remain = meow_grp[q:] + purr_grp[q:] + rest
        remain.sort(key=lambda d: -max(d['conf_meow'], d['conf_purr']))
        sel += remain[:TOP_K - len(sel)]
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(sel, f, ensure_ascii=False, indent=1)
    print(f'\n配额选样完成: {len(sel)} 条 (meow 主导 {min(len(meow_grp), q)}, '
          f'purr 主导 {min(len(purr_grp), q)})')
    print(f'输出: {OUT_JSON}')


if __name__ == '__main__':
    main()
