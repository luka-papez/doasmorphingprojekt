Ako koristite neki IDE obavezno dodati iznimke u .gitignore da se ne pusha
smece na server.

Za pokretanje je potrebno imati instalirano:
  1) OpenCV verziju barem 3.0.0. (ucitavanje i racunanje sa slikama)
      1) sve drugo potrebno za OpenCV je stavljeno vec u data folder jer ima
        samo par kilobajta
  2) dlib (detekcija tocaka lica)
      1) za dlib je potreban boost-python, nekad nije instaliran po defaultu
      2) potrebno je skinuti detektor lica dlib-a koji ima oko 100 mb
        i staviti ga u data folder. NI SLUCAJNO NE PUSHATI TO NA GIT!!
        dodano je da ga se ignorira u .gitignore ali provjeriti svaki put.	
        link ovdje (otpakirati, ocekuje se .dat fajl):
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Preporuka je koristiti Linux (moze u virtualci) jer se OpenCV tamo instalira
s par naredbi a instalacija dlib-a je doslovno samo "pip install dlib".

Upute za podesavanje OpenCV-a na Windowsu su ovdje:
http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/windows_install/windows_install.html

Kad se instalira sve, demo primjer se pokrene s:
python main.py -s images/conan.jpg -d ./images/governor.jpg
Nakon zavrsetka programa, u folderu morphs bit trebao biti spremljen
prijelaz izmedju slika. Sve slike u tom folderu se ignoriraju na gitu.
