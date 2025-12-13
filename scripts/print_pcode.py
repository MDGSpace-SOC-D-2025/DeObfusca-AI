# Ghidra headless script (Jython) to print P-code of the function "main"
# Place this in Ghidra's Scripts folder and run via headless analyzer or GUI.

from ghidra.app.script import GhidraScript

class PrintPcode(GhidraScript):
    def run(self):
        func = self.getFunctionAt(self.currentAddress)
        if func is None:
            funcs = list(self.getFunctions(True))
            # try to find a function named main
            for f in funcs:
                if f.getName() == "main":
                    func = f
                    break
            if func is None:
                self.println("No " + "main" + " found in current program")
                return
        self.println("Function: {}".format(func.getName()))
        pcode = self.decompileFunction(func)
        self.println(str(pcode))

# If run as script
if __name__ == '__main__':
    script = PrintPcode()
    script.run()
