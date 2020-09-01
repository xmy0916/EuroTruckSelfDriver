# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '1.8.4'
major           = '1'
minor           = '8'
patch           = '4'
rc              = '0'
istaged         = True
commit          = '1e01335e195d993f3c5c97bed3a15a6f9170acea'
with_mkl        = 'ON'

def show():
    if istaged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)

def mkl():
    return with_mkl
