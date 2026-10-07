"""
猫咪叫声情绪推理 (3 类语义输出)
================================
基于 YAMNet 微调最终模型 (finetune_final_hybrid 方案, 混合数据):
  跨域测试 Top-1 80% (4 类) / Macro F1 0.794
  calm/intense 组间语义 100% (resting+happy vs hunting+fighting 零泄漏)
  3 类语义: 安静 resting | 开心 happy | 激烈 intense (hunting+fighting 合并)

用法:
  python inference.py audio.wav
  python inference.py some_folder/ --model path/to/model.h5
  python inference.py audio.wav --raw        # 输出原始 4 类概率

依赖: tensorflow>=2.16, soundfile, resampy, numpy
模型: 由 experiments/finetune_final_hybrid.py 训练生成 (约 36MB)
"""
import os, sys, argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import soundfile as sf
import resampy
import tensorflow as tf

SAMPLE_RATE = 16000
FIXED_LEN   = SAMPLE_RATE * 6

RAW_CLASSES = ['resting', 'happy', 'hunting', 'fighting']
EMOTION_3   = {
    'resting':  ('安静', 'resting'),   # 休息/放松状态的声音
    'happy':    ('开心', 'happy'),     # 愉快/亲昵的叫声
    'hunting':  ('激烈', 'intense'),   # 狩猎/打斗等激烈叫声
    'fighting': ('激烈', 'intense'),
}


def patch_keras3_allow_slash():
    """Keras 3 禁止层名含 '/', 而官方 yamnet.h5 权重命名含 '/', 需打补丁放行"""
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


def build_inference_model(yamnet_dir):
    """重建与训练时完全一致的模型结构 (含微调权重), 返回 model"""
    patch_keras3_allow_slash()
    import keras
    sys.path.insert(0, str(yamnet_dir))
    import params as yamnet_params
    import yamnet as yamnet_model

    params = yamnet_params.Params()
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
    _, emb = yamnet_model.yamnet(x, params)
    emb = keras.layers.Lambda(
        lambda t: tf.reduce_mean(
            tf.reshape(t, (-1, n_patches, tf.shape(t)[-1])), axis=1),
        name='patch_mean')(emb)
    h = keras.layers.Dense(512, activation='relu', name='fc1')(emb)
    h = keras.layers.Dropout(0.3, name='drop1')(h)
    h = keras.layers.Dense(256, activation='relu', name='fc2')(h)
    h = keras.layers.Dropout(0.3, name='drop2')(h)
    out = keras.layers.Dense(4, activation='softmax', name='softmax')(h)
    return keras.Model(inputs=inp, outputs=out, name='inference')


def load_model(model_path, yamnet_dir):
    model = build_inference_model(yamnet_dir)
    model.load_weights(str(model_path))   # h5 权重按层名对齐, 不加载 Lambda config
    return model


def load_wav(path, target_sr=SAMPLE_RATE):
    wav_data, sr = sf.read(str(path), dtype=np.int16)
    wave = (wav_data / 32768.0).astype('float32')
    if wave.ndim > 1: wave = wave.mean(axis=1)
    if sr != target_sr: wave = resampy.resample(wave, sr, target_sr)
    if len(wave) < FIXED_LEN:                       # 短音频循环填充
        wave = np.tile(wave, FIXED_LEN // len(wave) + 1)[:FIXED_LEN]
    return wave[:FIXED_LEN]


def predict(model, wave):
    probs = model.predict(wave[None, :], verbose=0)[0]
    raw = {RAW_CLASSES[i]: float(probs[i]) for i in range(len(RAW_CLASSES))}
    agg = {'resting': raw['resting'], 'happy': raw['happy'],
           'intense': raw['hunting'] + raw['fighting']}
    top3 = max(agg, key=agg.get)
    cn = {'resting': '安静', 'happy': '开心', 'intense': '激烈'}[top3]
    return {'emotion_cn': cn, 'emotion_en': top3,
            'confidence': agg[top3], 'raw_probs': raw}


def main():
    ap = argparse.ArgumentParser(description='猫咪叫声情绪分析 (3 类语义)')
    ap.add_argument('input', help='wav 文件或目录')
    ap.add_argument('--model', default='models/yamnet_final_hybrid.h5',
                    help='模型路径 (默认 models/yamnet_final_hybrid.h5)')
    ap.add_argument('--yamnet-dir',
                    default=str(Path(__file__).resolve().parent / 'yamnet'),
                    help='yamnet 包目录 (含 yamnet.py / params.py)')
    ap.add_argument('--raw', action='store_true', help='额外输出原始 4 类概率')
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"模型不存在: {model_path}\n"
                 f"请先运行 experiments/finetune_final_hybrid.py 训练生成,\n"
                 f"或将已训练的 h5 放入 models/ 目录。")
    if not (Path(args.yamnet_dir) / 'yamnet.py').exists():
        sys.exit(f"yamnet 包不存在: {args.yamnet_dir}\n"
                 f"请下载 YAMNet 源码 (yamnet.py / params.py / yamnet_class_map.csv)\n"
                 f"放入该目录: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet")

    model = load_model(model_path, args.yamnet_dir)

    p = Path(args.input)
    files = sorted(p.glob('*.wav')) if p.is_dir() else [p]
    if not files:
        sys.exit(f"未找到 wav 文件: {p}")

    print(f"{'文件':<40} {'情绪':<8} {'置信度':>8}" + ("   原始4类概率" if args.raw else ""))
    print("-" * (60 if args.raw else 58))
    for f in files:
        try:
            r = predict(model, load_wav(f))
            line = f"{f.name:<40} {r['emotion_cn']}({r['emotion_en']})  {r['confidence']:>7.3f}"
            if args.raw:
                line += "   " + "  ".join(f"{k}={v:.3f}" for k, v in r['raw_probs'].items())
            print(line)
        except Exception as e:
            print(f"{f.name:<40} 错误: {e}")


if __name__ == '__main__':
    main()
