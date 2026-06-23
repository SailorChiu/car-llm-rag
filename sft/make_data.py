"""造车控 SFT 数据：自然语言 -> 严格 JSON 函数调用。

关键设计（对应数据审查 4 点）：
  1) 严格 schema 制造 base→SFT 差距：见 schema.py。
  2) 含难例：口语模糊（"有点闷"→开空调）、域外拒答（"今天天气"→system.reject）。
  3) train/val 按"措辞模板"切分，同一措辞绝不跨集 —— 验证集用训练里没出现过的说法，
     测的是 schema 泛化而不是背句子。
  4) 判分 schema 与序列化都来自 schema.py，训练/评测同一口径。
"""
import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from schema import to_json, REJECT_REASON  # noqa: E402

random.seed(7)

train, val = [], []


def emit(split, text, func, args):
    split.append({"messages_user": text, "func": func, "args": args,
                  "target": to_json(func, args)})


def numbers(lo, hi, k):
    """从 [lo,hi] 采样 k 个不同整数，控制样本量。"""
    pool = list(range(lo, hi + 1))
    random.shuffle(pool)
    return sorted(pool[:k])


# ===== climate.set_temperature（zone, value 16-32）=====
# 训练措辞
for v in numbers(16, 32, 10):
    emit(train, f"把温度调到{v}度", "climate.set_temperature", {"zone": "all", "value": v})
    emit(train, f"空调开到{v}度", "climate.set_temperature", {"zone": "all", "value": v})
for v in numbers(16, 32, 6):
    emit(train, f"主驾这边调到{v}度", "climate.set_temperature", {"zone": "driver", "value": v})
    emit(train, f"副驾温度设成{v}度", "climate.set_temperature", {"zone": "passenger", "value": v})
# 验证措辞（训练里没出现过的说法）
for v in numbers(16, 32, 6):
    emit(val, f"我想要{v}度", "climate.set_temperature", {"zone": "all", "value": v})
    emit(val, f"驾驶位弄到{v}度", "climate.set_temperature", {"zone": "driver", "value": v})

# ===== climate.set_power（on/off）=====
for t in ["打开空调", "开一下空调", "把空调打开"]:
    emit(train, t, "climate.set_power", {"state": "on"})
for t in ["关闭空调", "把空调关了", "空调关掉"]:
    emit(train, t, "climate.set_power", {"state": "off"})
for t in ["空调开起来", "来点冷气"]:
    emit(val, t, "climate.set_power", {"state": "on"})
for t in ["别开空调了", "不用空调了"]:
    emit(val, t, "climate.set_power", {"state": "off"})

# ===== climate.set_fan_speed（1-7）=====
for l in numbers(1, 7, 5):
    emit(train, f"风量调到{l}档", "climate.set_fan_speed", {"level": l})
    emit(train, f"把风开到{l}档", "climate.set_fan_speed", {"level": l})
for l in numbers(1, 7, 4):
    emit(val, f"风速设成{l}", "climate.set_fan_speed", {"level": l})

# ===== seat.set_heating（seat, level 0-3）=====
for l in range(0, 4):
    emit(train, f"主驾座椅加热开到{l}档", "seat.set_heating", {"seat": "driver", "level": l})
    emit(train, f"副驾座椅加热{l}档", "seat.set_heating", {"seat": "passenger", "level": l})
for l in [1, 3]:
    emit(val, f"驾驶位座椅加热调到{l}档", "seat.set_heating", {"seat": "driver", "level": l})
    emit(val, f"把副驾座椅加热设为{l}", "seat.set_heating", {"seat": "passenger", "level": l})

# ===== window.set_position（window, percent；0=关闭 100=全开）=====
wins = ["driver", "passenger", "rear_left", "rear_right"]
win_word = {"driver": "主驾", "passenger": "副驾", "rear_left": "左后", "rear_right": "右后"}
for w in wins:
    for p in numbers(10, 90, 2):
        emit(train, f"把{win_word[w]}车窗开到{p}%", "window.set_position", {"window": w, "percent": p})
emit(train, "关上所有车窗", "window.set_position", {"window": "all", "percent": 0})
emit(train, "打开主驾车窗", "window.set_position", {"window": "driver", "percent": 100})
for w in wins:
    p = random.choice(range(10, 91))
    emit(val, f"{win_word[w]}车窗降到{p}%", "window.set_position", {"window": w, "percent": p})
emit(val, "所有车窗都升上去", "window.set_position", {"window": "all", "percent": 0})

# ===== media.control =====
train_media = {"播放音乐": "play", "暂停": "pause", "下一首": "next", "上一首": "previous"}
val_media = {"放首歌听": "play", "先暂停一下": "pause", "切下一首": "next", "回到上一首": "previous"}
for t, a in train_media.items():
    emit(train, t, "media.control", {"action": a})
for t, a in val_media.items():
    emit(val, t, "media.control", {"action": a})

# ===== media.set_volume（0-100）=====
for v in numbers(0, 100, 8):
    emit(train, f"音量调到{v}", "media.set_volume", {"value": v})
    emit(train, f"声音开到{v}", "media.set_volume", {"value": v})
for v in numbers(0, 100, 5):
    emit(val, f"把音量设成{v}", "media.set_volume", {"value": v})

# ===== navigation.navigate_to =====
dests = ["最近的充电站", "公司", "火车站", "首都机场", "万达广场", "人民医院"]
for d in dests:
    emit(train, f"导航到{d}", "navigation.navigate_to", {"destination": d})
    emit(train, f"带我去{d}", "navigation.navigate_to", {"destination": d})
for d in dests:
    emit(val, f"去{d}怎么走", "navigation.navigate_to", {"destination": d})

# ===== phone.call =====
contacts = ["老婆", "张经理", "妈妈", "李医生", "客服"]
for c in contacts:
    emit(train, f"给{c}打电话", "phone.call", {"contact": c})
    emit(train, f"打电话给{c}", "phone.call", {"contact": c})
for c in contacts:
    emit(val, f"拨打{c}的电话", "phone.call", {"contact": c})

# ===== 难例：口语模糊（comfort）映射到明确动作 =====
for t in ["车里有点闷", "空气不太流通", "感觉有点憋"]:
    emit(train, t, "climate.set_power", {"state": "on"})
emit(val, "里面有点闷得慌", "climate.set_power", {"state": "on"})

# ===== 难例：域外/无法执行 -> system.reject =====
reject_train = ["今天天气怎么样", "讲个笑话", "明天股票会涨吗", "你叫什么名字",
                "帮我算一下 35 乘 28", "中国首都是哪里"]
reject_val = ["现在几点了", "推荐一部好看的电影", "帮我写封请假邮件", "世界杯谁夺冠了"]
for t in reject_train:
    emit(train, t, "system.reject", {"reason": REJECT_REASON})
for t in reject_val:
    emit(val, t, "system.reject", {"reason": REJECT_REASON})


# ============================================================
# 定向补例：对着首轮 SFT 真实错误补数据（只补 train 侧，措辞与 val 严格不重叠）
# ============================================================

# 错误A：climate.set_fan_speed 被误判成 set_power —— 训练没教过"风速"一词、且要带 level。
# 补多种"风力/风速/风量"措辞，模板均≠val 的"风速设成{l}"。
for l in range(1, 8):
    emit(train, f"风力调到{l}档", "climate.set_fan_speed", {"level": l})
    emit(train, f"风速开到{l}档", "climate.set_fan_speed", {"level": l})
for l in [1, 2, 3, 5, 7]:
    emit(train, f"空调风量开到{l}", "climate.set_fan_speed", {"level": l})

# 错误C：media.set_volume 参数名被串成 percent —— 强化"音量→value"，与 window 的 percent 对比。
for v in [5, 11, 30, 45, 55, 70, 82, 85, 95]:
    emit(train, f"音量设到{v}", "media.set_volume", {"value": v})
    emit(train, f"把声音弄到{v}", "media.set_volume", {"value": v})

# 错误D：window 左右混淆 + 升降方向错（升=关=0）。补左右对比 + 升/关=0 的方向样例。
for p in [20, 40, 60, 75, 90]:
    emit(train, f"右后车窗调到{p}%", "window.set_position", {"window": "rear_right", "percent": p})
    emit(train, f"左后那个车窗开到{p}%", "window.set_position", {"window": "rear_left", "percent": p})
emit(train, "把所有车窗升起来", "window.set_position", {"window": "all", "percent": 0})
emit(train, "关好所有车窗", "window.set_position", {"window": "all", "percent": 0})
emit(train, "升起左后车窗", "window.set_position", {"window": "rear_left", "percent": 0})
emit(train, "关闭右后车窗", "window.set_position", {"window": "rear_right", "percent": 0})

# 错误E：media.control 措辞太少（每 action 仅 1 条）。补多说法，均≠val 措辞。
mc_more = {"继续播放": "play", "放音乐吧": "play",
           "停一下音乐": "pause", "暂停播放": "pause",
           "下一曲": "next", "换下一首": "next", "跳到下一首": "next",
           "上一曲": "previous", "返回上一首": "previous"}
for t, a in mc_more.items():
    emit(train, t, "media.control", {"action": a})

# 错误B：system.reject 主题太窄（娱乐/体育知识被当成媒体指令）。补更多样的域外知识/闲聊。
reject_more = ["背一首古诗", "解释一下相对论", "明天会下雨吗", "帮我翻译这句英文",
               "NBA 昨天谁赢了", "奥运会几年一届", "最新有什么科幻片", "讲讲三国历史",
               "一公斤等于多少斤", "你最喜欢什么颜色"]
for t in reject_more:
    emit(train, t, "system.reject", {"reason": REJECT_REASON})


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    random.shuffle(train)
    random.shuffle(val)
    write(config.TRAIN_PATH, train)
    write(config.VAL_PATH, val)
    from collections import Counter
    tc = Counter(r["func"] for r in train)
    vc = Counter(r["func"] for r in val)
    print(f"train={len(train)} 条, val={len(val)} 条")
    print(f"{'function':<26}{'train':<8}val")
    print("-" * 42)
    for fn in sorted(set(tc) | set(vc)):
        print(f"{fn:<26}{tc.get(fn,0):<8}{vc.get(fn,0)}")
    print(f"\n已写出: {config.TRAIN_PATH}\n        {config.VAL_PATH}")


if __name__ == "__main__":
    main()
