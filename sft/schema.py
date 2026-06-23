"""车控指令的固定函数 schema —— make_data.py(造数据) 与 eval_sft.py(判分) 共用同一份。

设计意图：用 base 模型默认不会照搬的"严格 schema"制造 base→SFT 的可见差距：
  - 函数名固定为带域前缀的 snake_case（climate.set_temperature），base 常丢前缀或用中文/别名；
  - 参数名、类型、枚举值锁死；
  - 只输出紧凑 JSON（无多余文字），base 常带解释性废话；
  - 域外请求必须走 system.reject，base 几乎不会主动拒答。
"""
import json
import re

# function -> {参数名: 约束}。约束写法："int:lo-hi" / "str" / [枚举值...]
FUNCTIONS = {
    "climate.set_temperature": {"zone": ["driver", "passenger", "all"], "value": "int:16-32"},
    "climate.set_power":       {"state": ["on", "off"]},
    "climate.set_fan_speed":   {"level": "int:1-7"},
    "seat.set_heating":        {"seat": ["driver", "passenger"], "level": "int:0-3"},
    "window.set_position":     {"window": ["driver", "passenger", "rear_left", "rear_right", "all"],
                                "percent": "int:0-100"},
    "media.control":           {"action": ["play", "pause", "next", "previous"]},
    "media.set_volume":        {"value": "int:0-100"},
    "navigation.navigate_to":  {"destination": "str"},
    "phone.call":              {"contact": "str"},
    "system.reject":           {"reason": "str"},
}

# 域外/无法执行时统一的拒答理由（固定文本，训练/评测口径一致）
REJECT_REASON = "该请求超出车控指令范围"


def to_json(func, args):
    """规范输出：键顺序按 FUNCTIONS 定义、紧凑、保留中文。训练标签与判分都用它。"""
    ordered = {k: args[k] for k in FUNCTIONS[func] if k in args}
    return json.dumps({"function": func, "arguments": ordered},
                      ensure_ascii=False, separators=(",", ":"))


def extract_first_json(text):
    """从模型输出里抠出第一个完整 JSON 对象，解析失败返回 None。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def score(pred_text, gold_func, gold_args):
    """返回 (func_ok, exact_ok)。
    func_ok: 函数名完全一致；exact_ok: 函数名 + 参数字典都完全一致。
    """
    obj = extract_first_json(pred_text)
    if not isinstance(obj, dict):
        return False, False
    func_ok = obj.get("function") == gold_func
    exact_ok = func_ok and obj.get("arguments") == gold_args
    return func_ok, exact_ok


def _norm_fn(name):
    """函数名归一化到'末段'：取最后一个 '.' 后的部分，去掉非字母数字、转小写。
    例如 media.set_volume / set_volume / setVolume -> 'setvolume'。
    """
    seg = str(name).split(".")[-1]
    return re.sub(r"[^a-z0-9]", "", seg.lower())


def _norm_val(v):
    return str(v).strip().lower()


def score_intent(pred_text, gold_func, gold_args):
    """宽口径"意图准确率"：函数名只比末段(set_volume==media.set_volume)，
    参数只看 value 是否对得上(忽略参数名/大小写/markdown围栏/多余字段)。
    用来证明 base 其实'听懂了意图'、只是不合规，避免严格指标被质疑成稻草人。
    """
    obj = extract_first_json(pred_text)
    if not isinstance(obj, dict):
        return False
    if _norm_fn(obj.get("function")) != _norm_fn(gold_func):
        return False
    pred_args = obj.get("arguments", {})
    pred_vals = {_norm_val(x) for x in pred_args.values()} if isinstance(pred_args, dict) else set()
    gold_vals = {_norm_val(x) for x in gold_args.values()}
    return gold_vals.issubset(pred_vals)
