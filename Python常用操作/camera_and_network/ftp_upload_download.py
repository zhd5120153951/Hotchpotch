'''
@FileName   :ftp_upload_download.py
@Description:linux平台
@Date       :2022/09/22 15:58:58
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import ftplib


def download_file(filename):
    localfile = open(filename, 'wb')
    ftp.retrbinary('RETR ' + filename, localfile.write, 1024)

    ftp.quit()
    localfile.close()


def upload_file(filename):
    ftp.storbinary('STOR ' + filename, open(filename, 'rb'))
    ftp.quit()


"""
run this command in server first:
sudo apt install python3-pyftpdlib
sudo python3 -m pyftpdlib  -i xxx.xxx.x.x -p 2121 -w
"""

ftp = ftplib.FTP(host='10.10.1.111')
print('Connect')

ftp.login(user='weit', passwd='weit2.71')
print('Login')

ftp.cwd('/home/')

upload_file('test.txt'), print('Upload')
download_file('test.txt'), print('Download')
