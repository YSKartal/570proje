# 570proje

Pump It Up!

main.py dosyası çalıştırılır

kullandığımız tüm modeller farklı parametrelerle train 5fold cv sonuçları bastırılır

şu anda label encoding feature selecetion olmadan ayarlı ancak ######################## ile belirtilen alanlar ile encoding ve feature selection değiştirilebilir. one hot encoding seçildiğinde feature selection hata vermektedir, çünkü one hot dataframe memory sorunu oluşturduğu için sparse matrix kullanıyor ve bizim kurgumuzda feature selectionlar dataframe ile yapılabiliyor, o yüzden one hot tek başına kullanılmalı

en sonda ise submission yapmak istediğimiz modelin ayarlaması yapılarak sabmission dosyası oluşturulur. şu anda en başarılı yöntem sonuçları verilmekte, ancak label encoding değiştirilmezse
