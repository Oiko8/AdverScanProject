import os
from configs.settings import REPORT_DIR


def save_report(filename, content):
    """
    Saves a plain text report to the outputs directory.

    Args:
        filename : e.g. "week2_autoattack.txt"
        content  : string content to write
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"\n  Report saved to: {path}")