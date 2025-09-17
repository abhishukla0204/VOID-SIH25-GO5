import sys
print("Python path:", sys.path)
print("Current working directory:", __file__)

try:
    import main
    print("Main imported successfully")
    print("Main attributes:", dir(main))
    if hasattr(main, 'app'):
        print("App found!")
    else:
        print("App NOT found!")
except Exception as e:
    print("Error importing main:", e)
    import traceback
    traceback.print_exc()