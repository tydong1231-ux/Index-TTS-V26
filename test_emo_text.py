"""测试情感描述文本的实际8维向量输出"""
import sys, os, json, time

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "indextts"))

from indextts.infer_v2 import QwenEmotion

MODEL_DIR = os.path.join(APP_DIR, "checkpoints", "qwen0.6bemo4-merge")
print(f"Loading QwenEmotion from {MODEL_DIR}...")
qwen = QwenEmotion(MODEL_DIR)
print("Done.\n")

# 官方给的提示 + 从项目中找到的示例
# emo_text可以是任意自然语言
tests = [
    # === 官方UI提示 ===
    ("官方示例：委屈巴巴", "委屈巴巴"),
    ("官方示例：危险在悄悄逼近", "危险在悄悄逼近"),

    # === 单关键词 ===
    ("单关键词：高兴", "高兴"),
    ("单关键词：愤怒", "愤怒"),
    ("单关键词：悲伤", "悲伤"),
    ("单关键词：恐惧", "恐惧"),
    ("单关键词：惊喜", "惊喜"),
    ("单关键词：平静", "平静"),

    # === 程度修饰 ===
    ("程度修饰：极度愤怒", "极度愤怒"),
    ("程度修饰：轻微悲伤", "轻微悲伤"),

    # === 复合描述 ===
    ("复合描述：愤怒，咬牙切齿", "愤怒，咬牙切齿"),
    ("复合描述：悲伤，低沉哽咽", "悲伤，低沉哽咽"),
    ("复合描述：压抑、低落、说话很慢", "压抑、低落、说话很慢"),

    # === 场景叙事 ===
    ("场景叙事：他背叛了你，毁了你的一切", "他背叛了你，毁了你的一切"),
    ("场景叙事：外面下着大雨，你一个人坐在空房间里", "外面下着大雨，你一个人坐在空房间里"),
    ("场景叙事：你中了大奖，人生彻底改变", "你中了大奖，人生彻底改变"),

    # === 情感句子 ===
    ("情感句子：你吓死我了！你是鬼吗？", "你吓死我了！你是鬼吗？"),
    ("情感句子：哭哭。。。苦苦。。。", "哭哭。。。苦苦。。。"),

    # === 低落关键词（触发互换） ===
    ("低落互换：低落", "低落"),
    ("低落互换：我心情很低落", "我心情很低落"),
    ("低落互换：sad and depressed", "sad and depressed"),
]

HEADER = f"{'标签':<28} {'喜':>6} {'怒':>6} {'哀':>6} {'惧':>6} {'厌':>6} {'低':>6} {'惊':>6} {'平':>6}"
print("=" * 90)
print(HEADER)
print("-" * 90)

for label, text in tests:
    try:
        result = qwen.inference(text)
        # result is ordered dict: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
        vals = list(result.values())
        row = " ".join(f"{v:6.2f}" for v in vals)
        print(f"{label:<28} {row}")
    except Exception as e:
        print(f"{label:<28} ERROR: {e}")

print("-" * 90)
print("注：向量顺序 = 喜/怒/哀/惧/厌恶/低落/惊喜/平静")
print("    所有值经过 clamp [0, 1.2] 后再用于推理")
