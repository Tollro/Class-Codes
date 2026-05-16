# coding=utf-8
import io
import os
import re
import sys

import requests
from PIL import Image
from urllib import error

num = 0
numPicture = 0
save_dir = ""
List = []

# 浏览器 UA，降低被服务端拒绝的概率
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
    )
}


def save_valid_jpeg(content, path):
    """校验通过后转为 RGB 再保存为 JPEG，避免半损文件写入磁盘。"""
    try:
        im = Image.open(io.BytesIO(content))
        im.load()
        if im.mode in ("RGBA", "P", "LA"):
            im = im.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        im.save(path, "JPEG", quality=92)
        im.close()
        return True
    except Exception:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return False


def Find(base_url):
    global List
    print("正在检测图片总数，请稍等.....")
    t = 0
    s = 0
    while t < 1000:
        url = base_url + str(t) + "&gsm=8c"
        try:
            result = requests.get(url, timeout=7, headers=HEADERS)
        except BaseException:
            t = t + 60
            continue
        else:
            pic_url = re.findall('"objURL":"(.*?)",', result.text, re.S)
            s += len(pic_url)
            if len(pic_url) == 0:
                break
            List.append(pic_url)
            t = t + 60
    return s


def downloadPicture(html, keyword):
    global num
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    print("找到关键词:" + keyword + "的图片，即将开始下载图片...")
    for each in pic_url:
        if num >= numPicture:
            return
        print("正在尝试第" + str(num + 1) + "张，地址:" + str(each))
        try:
            if each is None:
                continue
            pic = requests.get(each, timeout=10, headers=HEADERS)
            if pic.status_code != 200 or not pic.content:
                print("跳过：HTTP 无内容或非 200")
                continue
        except BaseException:
            print("跳过：当前图片无法下载")
            continue

        safe_kw = re.sub(r'[\\/:*?"<>|]', "_", keyword)
        out_path = os.path.join(save_dir, "%s_%d.jpg" % (safe_kw, num))
        if not save_valid_jpeg(pic.content, out_path):
            print("跳过：损坏或无法识别的图片数据")
            continue

        num += 1
        print("已保存: " + out_path)


if __name__ == "__main__":
    default_kw = "芒果"
    raw = input("输入搜索关键词（直接回车默认「%s」）: " % default_kw).strip()
    word = raw if raw else default_kw

    base = (
        "http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word="
        + word
        + "&ct=201326592&v=flip"
    )
    tot = Find(base)
    print("检测到「%s」相关图片约 %d 张（估算）" % (word, tot))
    numPicture = int(input("输入想要下载的有效图片数量: "))
    save_dir = input("存放图片的文件夹（不存在则创建）: ").strip()
    if not save_dir:
        print("未指定目录，退出。")
        sys.exit(1)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    t = 0
    while num < numPicture and t < 1000:
        page_url = base + "&pn=" + str(t) + "&gsm=8c"
        try:
            result = requests.get(page_url, timeout=10, headers=HEADERS)
        except error.HTTPError:
            print("网络错误，请调整网络后重试")
        else:
            downloadPicture(result.text, word)
        t = t + 60

    print("完成。成功保存 %d 张有效图片到: %s" % (num, os.path.abspath(save_dir)))
