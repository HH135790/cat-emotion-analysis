# -*- coding: utf-8 -*-
"""最终模型训练 —— 混合数据 (wav1 选样 + 人工 hunting 数据)
================================================================
跨域测试 (25 条独立采集音频): Top-1 80% (16/20 in-scope), Macro F1 0.794
calm/intense 组间语义 100% (resting+happy vs hunting+fighting 零泄漏)
数据构成:
  resting : wav1 (data/train) 按猫叫置信度取 top-150
  happy   : wav1 配额选样 (75 meow 主导 + 75 purr 辅助, 见 select_happy_quota.py)
            —— 纯置信度 top-150 会把 happy 池选成 purr 集群, 导致 resting 泄漏
  fighting: wav1 全量 150
  hunting : data/manual_hunting/ 人工采集真实录音段 (质检后 189 段),
            不做响度归一化 —— 保留"轻 vs 响"能量判据
配方: 两阶段微调 (冻结练头 lr=1e-3 x12ep -> 解冻 layer12-14 lr=1e-4 <=30ep)
     + 增强 v2 六种确定性轮转 x2 -> 约 2300 条训练样本, seed 42
实现注意 (numba 病理性挂起规避):
  - soundfile 读音频; 变速/变调用 scipy resample_poly (非 librosa)
输出: output/yamnet_final_hybrid.h5 + report.json + test_predictions.csv
"""
import os, sys, csv, json
from pathlib import Path
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfiltfilt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

ROOT         = Path(__file__).resolve().parents[1]   # 仓库根目录
YAMNET_PKG   = ROOT / 'yamnet'                        # 官方 YAMNet 包 (自行下载)
LABELS_CSV   = ROOT / 'data' / 'labels_train.csv'     # filename, v2_emotion, ...
CONF_CSV     = ROOT / 'data' / 'mapping.csv'          # filename, confidence (detect_cat_sounds.py 生成)
WAV1_DIR     = ROOT / 'data' / 'train'
HUNT_DIR     = ROOT / 'data' / 'manual_hunting'       # 人工采集 hunting 段 (真实录音域)
TEST_DIR     = ROOT / 'data' / 'test'
TEST_LABELS  = ROOT / 'data' / 'labels_test.csv'
OUT_DIR      = ROOT / 'output'

CLASSES  = ['resting', 'happy', 'hunting', 'fighting']
CLS2IDX  = {c: i for i, c in enumerate(CLASSES)}
TOP_K    = 150
AUG_PER  = 2
SR       = 16000
FIXED_LEN = SR * 6
SEED     = 42


def patch_keras3_allow_slash():
    """Keras 3 不允许层名含 '/', 官方 yamnet.h5 权重命名含 '/', 打补丁放行"""
    import keras.src.ops.operation as _op_mod
    def _patched_op_init(self, name=None, *a, **kw):
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
    _op_mod.Operation.__init__ = _patched_op_init
    import keras.src.backend.common.name_scope as _ns_mod
    def _patched_ns_init(self, name, caller=None, deduplicate=True, override_parent=None):
        if not isinstance(name, str): raise ValueError('name must be str')
        self.name = name; self.caller = caller; self.deduplicate = deduplicate
        self.override_parent = override_parent
        if override_parent is None and deduplicate and getattr(caller, '_parent_path', None) is not None:
            self.override_parent = caller._parent_path
        self._pop_on_exit = False
    _ns_mod.name_scope.__init__ = _patched_ns_init


def load_wav(path):
    y, native = sf.read(str(path), dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    if native != SR:
        g = gcd(int(native), SR)
        y = resample_poly(y, SR // g, int(native) // g).astype('float32')
    return y


def to_fixed_len(wave, length=FIXED_LEN):
    if len(wave) < length:
        return np.tile(wave, length // len(wave) + 1)[:length]
    return wave[:length]


def speed_pitch(wave, rate):
    """重采样变速变调 (scipy): rate>1 变快音高升高"""
    g = gcd(int(round(rate * 1000)), 1000)
    y = resample_poly(wave, 1000 // g, int(round(rate * 1000)) // g)
    return y.astype('float32')


N_AUG_KINDS = 6

def augment_v2(wave, rng, kind):
    """增强v2: 与基线同 6 种轮转; 1/2/5 的变速实现换 scipy (语义一致)"""
    if kind == 0:
        return wave + rng.normal(0, 0.01, wave.shape).astype('float32')
    if kind == 1:
        return speed_pitch(wave, float(rng.uniform(0.85, 1.15)))
    if kind == 2:
        return speed_pitch(wave, 2 ** (float(rng.uniform(-2, 2)) / 12))
    if kind == 3:
        gain = float(rng.uniform(0.5, 1.5))
        return (wave * gain + rng.normal(0, 0.003, wave.shape)).astype('float32')
    if kind == 4:
        cutoff = float(rng.uniform(800, 3000))
        spec = np.fft.rfft(wave)
        freqs = np.fft.rfftfreq(len(wave), 1.0 / SR)
        if rng.random() < 0.5:
            mask = (freqs <= cutoff).astype(np.float32)
        else:
            mask = (freqs >= cutoff).astype(np.float32)
        mask = np.convolve(mask, np.ones(51, dtype=np.float32) / 51, mode='same')
        return np.fft.irfft(spec * mask, n=len(wave)).astype('float32')
    w = speed_pitch(wave, float(rng.uniform(0.9, 1.1)))
    return (w + rng.normal(0, 0.005, w.shape)).astype('float32')


def build_selection():
    """resting top-150 置信度; happy 配额选样 (75 meow+75 purr, 复用 quota 实验 JSON);
    fighting 全量150; hunting 人工数据"""
    conf = {}
    with open(CONF_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try: conf[r['filename']] = float(r['confidence'])
            except (ValueError, KeyError): pass
    pool = {c: [] for c in ('resting', 'fighting')}
    with open(LABELS_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['v2_emotion'] in pool and r['filename'] in conf:
                pool[r['v2_emotion']].append((r['filename'], conf[r['filename']]))
    selection = {}
    pool['resting'].sort(key=lambda t: -t[1])
    selection['resting'] = [(WAV1_DIR / fn, 'resting') for fn, _ in pool['resting'][:TOP_K]]
    print(f"  resting    可用 {len(pool['resting'])}, 取 top-150", flush=True)
    # happy: 配额选样 (75 meow 主导 + 75 purr 辅助), 由 select_happy_quota.py 生成
    with open(ROOT / 'data' / 'happy_quota_selection.json', encoding='utf-8') as f:
        happy_list = json.load(f)
    selection['happy'] = [(Path(h['path']), 'happy') for h in happy_list]
    print(f"  happy      配额选样 {len(happy_list)} (75 meow + 75 purr)", flush=True)
    selection['fighting'] = [(WAV1_DIR / fn, 'fighting') for fn, _ in pool['fighting']]
    print(f"  fighting   可用 {len(pool['fighting'])}, 全量", flush=True)
    hunt_files = sorted(HUNT_DIR.glob('*.wav'))
    selection['hunting'] = [(p, 'hunting') for p in hunt_files]
    print(f"  hunting    人工采集 {len(hunt_files)} 段, 全量 (wav1 hunting 不参与)", flush=True)
    return selection


def build_train_set(selection):
    rng = np.random.default_rng(SEED)
    X, y = [], []
    counter = 0
    for c in CLASSES:
        n_ok = 0
        for fp, _ in selection[c]:
            if not fp.exists():
                print(f"  缺失: {fp.name}"); continue
            wave = to_fixed_len(load_wav(fp))
            X.append(wave); y.append(CLS2IDX[c])
            for j in range(AUG_PER):
                kind = (counter * AUG_PER + j) % N_AUG_KINDS
                X.append(to_fixed_len(augment_v2(wave, rng, kind)).astype('float32'))
                y.append(CLS2IDX[c])
            counter += 1; n_ok += 1
        print(f"  {c}: {n_ok} 原始 x{1+AUG_PER} = {n_ok*(1+AUG_PER)}", flush=True)
    return np.array(X, dtype='float32'), np.array(y)


def build_model(params, yamnet_lib):
    from tensorflow import keras
    win = int(round(params.sample_rate * params.stft_window_seconds))
    hop = int(round(params.sample_rate * params.stft_hop_seconds))
    fft = 2 ** int(np.ceil(np.log(win) / np.log(2.0)))
    bins = fft // 2 + 1
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.mel_bands, num_spectrogram_bins=bins,
        sample_rate=params.sample_rate,
        lower_edge_hertz=params.mel_min_hz, upper_edge_hertz=params.mel_max_hz)
    pw = int(round((params.sample_rate / hop) * params.patch_window_seconds))
    ph = int(round((params.sample_rate / hop) * params.patch_hop_seconds))
    n_frames  = (FIXED_LEN - win) // hop + 1
    n_patches = (n_frames - pw) // ph + 1

    def _feat(wave):
        mag = tf.abs(tf.signal.stft(wave, frame_length=win, frame_step=hop, fft_length=fft))
        mel = tf.matmul(mag, mel_w)
        lm = tf.math.log(mel + params.log_offset)
        patches = tf.signal.frame(lm, pw, ph, axis=1)
        return tf.reshape(patches, (-1, params.patch_frames, params.patch_bands))

    inp = keras.Input(shape=(FIXED_LEN,), dtype='float32', name='waveform')
    x = keras.layers.Lambda(_feat, name='logmel_patches')(inp)
    _, emb = yamnet_lib.yamnet(x, params)
    emb = keras.layers.Lambda(
        lambda t: tf.reduce_mean(
            tf.reshape(t, (-1, n_patches, tf.shape(t)[-1])), axis=1),
        name='patch_mean')(emb)
    h = keras.layers.Dense(512, activation='relu', name='fc1')(emb)
    h = keras.layers.Dropout(0.3, name='drop1')(h)
    h = keras.layers.Dense(256, activation='relu', name='fc2')(h)
    h = keras.layers.Dropout(0.3, name='drop2')(h)
    out = keras.layers.Dense(len(CLASSES), activation='softmax', name='softmax')(h)
    model = keras.Model(inputs=inp, outputs=out, name='final_hybrid')

    import h5py
    bb_map = {l.name: l for l in model.layers if l.weights}
    slot_order = {'kernel': 0, 'bias': 1, 'beta': 0, 'moving_mean': 1, 'moving_variance': 2}
    by_layer = {}
    with h5py.File(str(YAMNET_PKG / 'yamnet.h5'), 'r') as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                p = name.rsplit('/', 1)[0]
                c = p.split('/')
                if len(c) % 2 == 0 and c[:len(c)//2] == c[len(c)//2:]:
                    lname = '/'.join(c[:len(c)//2])
                    slot = name.rsplit('/', 1)[1].replace(':0', '')
                    by_layer.setdefault(lname, []).append((slot_order.get(slot, 9), obj[()]))
        f.visititems(_visit)
    n = 0
    for lname, arrs in by_layer.items():
        if lname in bb_map:
            arrs.sort(key=lambda t: t[0])
            ws = [np.asarray(a) for _, a in arrs]
            layer = bb_map[lname]
            if len(ws) == len(layer.weights) and all(w.shape == k.shape for w, k in zip(ws, layer.weights)):
                layer.set_weights(ws); n += 1
    print(f"权重迁移: {n}/{len(bb_map)} 层", flush=True)
    assert n >= 50, '权重迁移异常'
    return model


def set_freeze(model, phase):
    for l in model.layers:
        if l.name.startswith('layer'):
            block = int(l.name.split('/')[0].replace('layer', ''))
            l.trainable = (phase == 2 and block >= 12)
        else:
            l.trainable = True


def main():
    patch_keras3_allow_slash()
    sys.path.insert(0, str(YAMNET_PKG))
    import params as yamnet_params
    import yamnet as yamnet_model
    from tensorflow import keras

    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print(" 最终模型训练: 混合数据 (wav1 选样 + 人工 hunting, happy=配额)")
    print("=" * 70, flush=True)

    print("\n构建数据池:")
    selection = build_selection()
    print("\n加载音频 + 增强 v2 (scipy 变速)...")
    X, y = build_train_set(selection)
    idx = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[idx], y[idx]
    print(f"训练样本: {len(X)}", flush=True)

    params = yamnet_params.Params()
    model = build_model(params, yamnet_model)

    set_freeze(model, phase=1)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("\n--- 阶段 1: 冻结 YAMNet, lr=1e-3, 12 epochs ---", flush=True)
    model.fit(X, y, epochs=12, batch_size=32, verbose=2)

    set_freeze(model, phase=2)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("\n--- 阶段 2: 解冻 layer12-14, lr=1e-4, <=30 epochs ---", flush=True)
    model.fit(X, y, epochs=30, batch_size=32, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

    model.save(str(OUT_DIR / 'yamnet_final_hybrid.h5'))
    probs_tr = model.predict(X, verbose=0)
    acc_tr = float((probs_tr.argmax(axis=1) == y).mean())
    print(f"\n训练集自评: acc={acc_tr:.4f}", flush=True)

    # V2 评估
    test_labels_map = {r['filename']: r['v2_emotion']
                       for r in csv.DictReader(open(TEST_LABELS, encoding='utf-8-sig'))}
    X_te, te_true, te_names = [], [], []
    for fp in sorted(TEST_DIR.glob('*.wav')):
        X_te.append(to_fixed_len(load_wav(fp)))
        te_true.append(test_labels_map.get(fp.name, 'UNKNOWN'))
        te_names.append(fp.name)
    X_te = np.array(X_te, dtype='float32')
    probs = model.predict(X_te, verbose=0)
    preds = probs.argmax(axis=1)

    correct, in_scope = 0, 0
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    ood_flow = {}
    rows_out = []
    for i, tl in enumerate(te_true):
        pl = CLASSES[preds[i]]
        if tl in CLASSES:
            in_scope += 1
            if pl == tl: correct += 1
            cm[CLS2IDX[tl], preds[i]] += 1
        else:
            ood_flow.setdefault(tl, {}).setdefault(pl, 0)
            ood_flow[tl][pl] = ood_flow[tl][pl] + 1
        rows_out.append({
            'filename': te_names[i], 'true_label': tl, 'pred_label': pl,
            'in_scope': tl in CLASSES, 'match': pl == tl,
            'confidence': round(float(probs[i].max()), 4),
            **{f'p_{c}': round(float(probs[i, CLS2IDX[c]]), 4) for c in CLASSES},
        })

    scope_acc = correct / in_scope if in_scope else 0
    per_class, f1s = {}, []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        p = tp / cm[:, i].sum() if cm[:, i].sum() else 0
        r = tp / cm[i].sum() if cm[i].sum() else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        f1s.append(f1)
        per_class[c] = {'precision': round(p,3), 'recall': round(r,3),
                        'f1': round(f1,3), 'support': int(cm[i].sum())}
    macro_f1 = float(np.mean(f1s))

    print("\n" + "=" * 70)
    print(" 最终结果: 跨域测试 (独立采集音频)")
    print("=" * 70)
    print(f"Top-1: {correct}/{in_scope} = {scope_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    for c, s in per_class.items():
        print(f"  {c:<10} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} (n={s['support']})")
    print("\n混淆矩阵 (行=真值, 列=预测):")
    print("             " + " ".join(f"{c:<10}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:<10} " + " ".join(f"{cm[i,j]:<10}" for j in range(len(CLASSES))))
    print(f"\nOOD 流向 (非4类测试文件的预测去向): {ood_flow}", flush=True)

    with open(OUT_DIR / 'test_predictions.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

    summary = {
        'timestamp': '2026-10-06',
        'method': 'final hybrid: resting top-150 + happy quota(75 meow/75 purr) + fighting 150 '
                  'from wav1; hunting = 189 manual chatter segments (QC-passed, real-domain, '
                  'no loudness normalization); two-stage finetune + augv2 rotation; '
                  'scipy speed/pitch (numba-free)',
        'classes': CLASSES,
        'train_raw': {c: len(selection[c]) for c in CLASSES},
        'train_samples': int(len(X)),
        'train_acc': round(acc_tr, 4),
        'test_in_scope': in_scope,
        'test_top1': round(scope_acc, 4),
        'macro_f1': round(macro_f1, 4),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'ood_flow': ood_flow,
        'model_path': str(OUT_DIR / 'yamnet_final_hybrid.h5'),
    }
    with open(OUT_DIR / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {OUT_DIR / 'report.json'}")


if __name__ == '__main__':
    main()

