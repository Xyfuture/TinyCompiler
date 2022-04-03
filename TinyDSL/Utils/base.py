import collections

class linkNode:
    def __init__(self,value,**kwargs):
        self.pre = None
        self.next = None
        self.key = None

        self.value = value

        if 'pre' in kwargs:
            self.pre = kwargs['pre']
        if 'next' in kwargs:
            self.next = kwargs['next']
        if 'key' in kwargs:
            self.key = kwargs['key']

    def insert_pre(self,value,**kwargs):
        tmp = linkNode(value,**kwargs)
        tmp.pre = self.pre
        tmp.next = self
        self.pre = tmp

    def insert_next(self,value,**kwargs):
        tmp = linkNode(value,**kwargs)
        tmp.next = self.next
        tmp.pre = self
        self.next = tmp

    def __next__(self):
        if self.value is None:
            raise StopIteration
        return self.next





class linkList:
    def __init__(self):
        self.head = linkNode(value=None,key='Head')
        self.tail = linkNode(value=None,key='Tail')

        self.head.next = self.tail
        self.tail.pre = self.head

    def append(self,value,**kwargs):
        self.tail.insert_pre(value,**kwargs)

    def __str__(self):
        cur = self.head.next
        tmp = ""
        while cur is not self.tail:
            tmp += cur.value.__str__() + "\n"
        return tmp

    def __iter__(self):
        if self.head.next is self.tail:
            raise StopIteration
        return self.head.next


class bitmap:
    def __init__(self,num,default=False,free_state=False):
        self.num = num
        self.default = default
        self.free_state = free_state
        self.state = [default for i in range(self.num)]

    def query(self,i):
        assert 0 <= i < self.num, "ERROR: overflow"
        return self.state[i]

    def get_free(self,cnt=1,new_state=True,**kwargs):
        if cnt == 1:
            for i,s in enumerate(self.state):
                if s is self.free_state:
                    if not kwargs.get('unset'):
                        self.state[i] = new_state
                    return i
            raise "ERROR: no free position"
        else:
            tmp = [self.get_free(1) for _ in range(cnt)]
            return tmp

    def free(self,item):
        self.state[item] = self.free_state

    def reset(self):
        for i in range(self.num):
            self.state[i] = self.free_state

    def __getitem__(self, item):
        return self.state[item]

    def __setitem__(self, key, value):
        self.state[key] = value



