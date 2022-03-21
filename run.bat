CHCP 65001
echo ***************** The preprocess is doing*************************************************************************************************
cd D:\电子科大文件夹\大创\代码\MIAT_ECG-main
start python my_process_for_MIAT.py
Rem
echo ***************** The GAN is training for class =N *************************************************************************************************
cd GAN
start python GAN.py  --class=0
echo ***************** The GAN is training for class =S *************************************************************************************************
Rem
pause
start python GAN.py  --class=1
echo ***************** The GAN is training for class =V *************************************************************************************************
Rem
pause
start python GAN.py  --class=2
echo ***************** The GAN is training for class =F *************************************************************************************************
Rem
pause
start python GAN.py  --class=3
cd ..
Rem
pause
echo  ***************** The CNN-LSTM network is training *************************************************************************************************
start python MIAT_train_eval.py --run_id=0 --gpu=0 --epochs=150 --lr=0.001 --weight=0.005 --n=5 --lambda=1 --alpha=2 --prevat=1 --mix=1 --vat=1 --s=DS1 --t=DS2 --fake=True
Rem
exit