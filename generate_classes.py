
class Classes:
    def __init__(self, lower_bound, upper_bound):
        self.classes = {}
        for step in range(lower_bound, upper_bound - 10, 10):
            class_key = str(step) + ':' + str(step + 10)
            self.classes[class_key] = (step, step + 10)
    
    def getClass(self, val):
        for class_key in self.classes:
            lower_bound, upper_bound = self.classes[class_key]
            if lower_bound <= val and val < upper_bound:
                return class_key
        return -1


test = Classes(-240, 140)
print(test.getClass(-300))