"""
YAMNet 猫叫检测: 对目录中所有音频计算"是否为猫叫"及最高置信度, 输出 mapping.csv
================================================================================
原理: Google YAMNet 输出 521 类 AudioSet 标签, 其中 6 类与猫相关:
      Cat / Purr / Meow / Hiss / Caterwaul / Roaring cats
对每个文件取所有时间帧上猫类别的最高分作为该文件的猫叫置信度。
该置信度只回答"是不是猫叫", 不区分情绪。

用法:
  python detect_cat_sounds.py --input wav/ --output mapping.csv
"""
import os, sys, csv, argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import soundfile as sf
import resampy
import tensorflow as tf

SAMPLE_RATE = 16000
# AudioSet 中与猫相关的类别 display_name
CAT_DISPLAY_NAMES = ['Cat', 'Purr', 'Meow', 'Hiss', 'Caterwaul', 'Roaring cats']


def load_yamnet(yamnet_dir):
    """构建 Keras 3 兼容的批量 YAMNet 模型并加载官方权重, 返回 (model, params, cat_indices)

    官方 yamnet_frames_model 的 pad_waveform 在 Keras 3 下不兼容 (tf.shape(KerasTensor)),
    这里用与 finetune 脚本一致的批量结构: (B, 96000) -> STFT -> patches -> 核心 yamnet
    -> predictions (B*P, 521) -> reshape (B, P, 521)
    """
    import keras
    sys.path.insert(0, str(yamnet_dir))
    import params as yamnet_params
    import yamnet as yamnet_model

    def patch_keras3_allow_slash():
        import keras.src.ops.operation as _op_mod
        def _patched_op_init(self, name=None, *a, **kw):
            if name is None:
                import keras.src.utils.naming as _nm
                name = _nm.auto_name(self.__class__.__name__)
            if not isinstance(name, str): raise ValueError(f"name must be str, got {name}")
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
            if not isinstance(name, str): raise ValueError(f"name must be str, got {name}")
            self.name = name; self.caller = caller; self.deduplicate = deduplicate
            self.override_parent = override_parent
            if override_parent is None and deduplicate and getattr(caller, '_parent_path', None) is not None:
                self.override_parent = caller._parent_path
            self._pop_on_exit = False
        _ns_mod.name_scope.__init__ = _patched_ns_init

    patch_keras3_allow_slash()
    params = yamnet_params.Params()

    fixed_len = int(params.sample_rate * 6)   # 统一 6 秒输入
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
    n_frames  = (fixed_len - win) // hop + 1
    n_patches = (n_frames - pw) // ph + 1

    def _feat(wave):
        mag = tf.abs(tf.signal.stft(wave, frame_length=win, frame_step=hop, fft_length=fft))
        mel = tf.matmul(mag, mel_w)
        lm = tf.math.log(mel + params.log_offset)
        patches = tf.signal.frame(lm, pw, ph, axis=1)
        return tf.reshape(patches, (-1, params.patch_frames, params.patch_bands))

    inp = keras.Input(shape=(fixed_len,), dtype='float32', name='waveform')
    x = keras.layers.Lambda(_feat, name='logmel_patches')(inp)
    preds, _ = yamnet_model.yamnet(x, params)            # (B*P, 521)
    preds = keras.layers.Lambda(
        lambda t: tf.reshape(t, (-1, n_patches, tf.shape(t)[-1])),
        name='frame_reshape')(preds)                      # (B, P, 521)
    model = keras.Model(inputs=inp, outputs=preds, name='yamnet_frames_batched')
    model.load_weights(str(Path(yamnet_dir) / 'yamnet.h5'))

    # 从官方 class_map_csv 找到猫类别的输出索引
    cat_indices = []
    class_map = Path(yamnet_dir) / 'yamnet_class_map.csv'
    with open(class_map, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['display_name'] in CAT_DISPLAY_NAMES:
                cat_indices.append(int(row['index']))
    print(f"猫相关类别索引: {cat_indices}")
    return model, params, cat_indices


def cat_confidence(model, params, cat_idx, wav_path):
    """返回该文件所有帧上猫类别的最高分"""
    wav_data, sr = sf.read(str(wav_path), dtype=np.int16)
    wave = (wav_data / 32768.0).astype('float32')
    if wave.ndim > 1: wave = wave.mean(axis=1)
    if sr != SAMPLE_RATE: wave = resampy.resample(wave, sr, SAMPLE_RATE)
    if len(wave) == 0: return None
    fixed_len = int(params.sample_rate * 6)
    if len(wave) < fixed_len:
        wave = np.tile(wave, fixed_len // len(wave) + 1)[:fixed_len]
    wave = wave[:fixed_len]
    scores = model(wave[None, :], training=False).numpy()   # (1, P, 521)
    return float(scores[0][:, cat_idx].max())


def main():
    ap = argparse.ArgumentParser(description='YAMNet 猫叫检测 -> mapping.csv')
    ap.add_argument('--input', required=True, help='wav 目录')
    ap.add_argument('--output', default='mapping.csv', help='输出 csv 路径')
    ap.add_argument('--yamnet-dir', default=str(Path(__file__).resolve().parent.parent / 'yamnet'),
                    help='yamnet 包目录 (含 yamnet.h5 / yamnet_class_map.csv)')
    args = ap.parse_args()

    model, params, cat_idx = load_yamnet(args.yamnet_dir)
    files = sorted(Path(args.input).glob('*.wav'))
    print(f"共 {len(files)} 个文件")

    rows = []
    for i, fp in enumerate(files, 1):
        try:
            conf = cat_confidence(model, params, cat_idx, fp)
            rows.append({'filename': fp.name, 'is_cat': conf is not None and conf >= 0.3,
                         'confidence': round(conf, 4) if conf is not None else '',
                         'error': ''})
        except Exception as e:
            rows.append({'filename': fp.name, 'is_cat': '', 'confidence': '', 'error': str(e)})
        if i % 200 == 0:
            print(f"  {i}/{len(files)}")

    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'is_cat', 'confidence', 'error'])
        w.writeheader(); w.writerows(rows)

    ok = sum(1 for r in rows if r['is_cat'] == True)
    errs = sum(1 for r in rows if r['error'])
    print(f"完成: 猫叫 {ok}/{len(rows)}, 出错 {errs} -> {args.output}")


if __name__ == '__main__':
    main()
