from captcha.image import ImageCaptcha
import random
import sys
import os


class CodeCreator(object):
    def __init__(self, root, num, extension='jpg', charset='l'):
        """
        :param num: number of images to be generated
        :param root: root of the images to be saved to
        :param extension: type of the images, jpg or png
        :param charset: mode of the code, lower or upper or lower and upper
        """
        self.charset = [str(ii) for ii in range(0, 10)]
        self.root = root
        self.type = extension
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if charset == 'l':
            self.charset += [chr(ii) for ii in range(97, 97 + 26)]
        elif charset == 'u':
            self.charset += [chr(ii) for ii in range(65, 65 + 26)]
        elif charset == 'lu':
            self.charset += [chr(ii) for ii in range(97, 97 + 26)]
            self.charset += [chr(ii) for ii in range(65, 65 + 26)]
        else:
            assert charset == 'l' or charset == 'u' or charset == 'lu'
        for ii in range(num):
            self.generator()
            sys.stdout.write('\r>> Creating image %d/%d' % (ii + 1, num))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        print("生成完毕")

    def generator(self):
        image = ImageCaptcha()
        text = ''.join([random.choice(self.charset) for _ in range(4)])
        image.write(text, os.path.join(self.root, text + '.{}'.format(self.type)))


CodeCreator('image', 50000)
