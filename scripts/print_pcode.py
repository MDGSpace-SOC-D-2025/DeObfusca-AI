
from ghidra.app.script import GhidraScript

class PrintPcode(GhidraScript):
    def run(self):
        func = self.getFunctionAt(self.currentAddress)
        if func is None:
            funcs = list(self.getFunctions(True))
            for f in funcs:
                if f.getName() == "main":
                    func = f
                    break
            if func is None:
                self.println("No main found in current program")
                return
        self.println("Function: {}".format(func.getName()))
        pcode = self.decompileFunction(func)
        self.println(str(pcode))

if __name__ == '__main__':
    script = PrintPcode()
    script.run()
