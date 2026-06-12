"""ShellBench LSTM policy plugin for LeRobot.

`import policies.lstm.policy` 即完成註冊與 factory patch（透過 register 模組）。
"""

from .register import LSTMConfig, LSTMPolicy, make_lstm_pre_post_processors

__all__ = ["LSTMConfig", "LSTMPolicy", "make_lstm_pre_post_processors"]
