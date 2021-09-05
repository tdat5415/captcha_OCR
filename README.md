### captcha 글자 해독하기

---

다음 명령어로 captcha 이미지를 다운해준다.

```python
!wget https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
!unzip -q captcha_images_v2.zip
```



```python
import os
from glob import glob

img_paths = sorted(glob('./captcha_images_v2/*.png'))
get_label = lambda x:os.path.splitext(os.path.basename(x))[0]
labels = list(map(get_label, img_paths))
chars = set(''.join(labels))
max_length = max([len(label) for label in labels])

print("Number of images found: ", len(img_paths))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(chars))
print("Characters present: ", chars)
```
```
Number of images found:  1040
Number of labels found:  1040
Number of unique characters:  19
Characters present:  {'x', 'd', '2', '8', 'b', 'w', 'p', 'n', 'y', 'c', 'e', '4', 'f', '5', 'g', '7', '3', '6', 'm'}
```

---

참고 URL : 
<https://keras.io/examples/vision/captcha_ocr/>
<https://hulk89.github.io/machine%20learning/2018/01/30/ctc/>
<https://www.kaggle.com/fournierp/captcha-version-2-images>
