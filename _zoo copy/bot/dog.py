
if __name__=='__main__':
    from animal import animal
else:
    from .animal import animal 


class dog(animal):
    def __init__(self) -> None:
        super().__init__('Mammal')
        pass
    
    
    
    
    
    
    
if __name__=='__main__':
    d=dog()
    print(d.kind)