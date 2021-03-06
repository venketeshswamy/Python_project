 
# import kivy module     
import kivy   
import os
import subprocess     
from kivy.app import App
kivy.require('1.9.0') 

from kivy.lang import Builder 
from kivy.uix.screenmanager import ScreenManager, Screen    
lf=Builder.load_file("cwd.kv")   
class ScreenOne(Screen): 
    pass
   
class ScreenTwo(Screen): 
    pass
screen_manager = ScreenManager() 
screen_manager.add_widget(ScreenOne(name ="screen_one")) 
screen_manager.add_widget(ScreenTwo(name ="s2")) 
  
# Create the App class 
class main(App): 
    def build(self): 
        return screen_manager
    abspathtofile="Nothing till now"
    wd=os.getcwd()
    os=__import__("os")
    def changetoapp(self):
        os.startfile("main.py")
    def copytolocal(self,source):
        import os
        import shutil
        ext=source.split(".")[-1]
        print("EXT",ext)
        print("SRC",source)
        fn=source.split("\\")[-1]
        print("FN",fn)
        src=open(source,"rb")
        dst=open("User Content/{}".format(fn),"wb")
        dst.write(src.read())
# run the app  
main().run() 
