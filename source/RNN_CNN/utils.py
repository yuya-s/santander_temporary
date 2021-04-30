import numpy as np

class ListDict(dict):
    #dict型を継承して，valueがlist型の場合の面倒な処理をメソッド化
    def __init__(self,d={}):
        dict.__init__(self,d)
        
    def append(self,key,elem):
        if key in self.keys():
            self[key].append(elem)
        else:
            self[key] = [elem]
            
    def extend(self,key,target):
        if key in self.keys():
            self[key].extend(target)
        else:
            self[key] = [target]
            
def data_generator(data,timestamps,window_size = 48,slide =1,batch_size=1024,out_size=8):
    #月・日・曜日・時間を[0,1]で正規化
    standard_time = np.array(
        [
            (
                date.month/12,
                date.day/31,
                date.weekday()/7,
                (date.hour+(date.minute/60))/24
            )
            for date in timestamps
        ]
    )
    while(1):
        for i in range(0,len(data) -  window_size - out_size-batch_size-1,slide * batch_size):
            try:
                yield (
                    [
                        np.array([data[j:j+window_size] for j in range(i,i+batch_size * slide,slide)]),
                        np.array([standard_time[j+window_size+1:j+window_size+1+out_size] for j in range(i,i+batch_size * slide,slide)])
                    ],
                    np.array([data[j+window_size+1:j+window_size+1+out_size] for j in range(i,i+batch_size * slide,slide)]),
                )
            except:
                break

def data_generator_tl(data,window_size = 48,slide =1,batch_size=256,out_size=8):
    while(1):
        for i in range(0,len(data) -  window_size - out_size-batch_size-1,slide * batch_size):
            try:
                yield (
                    np.array([data[j:j+window_size] for j in range(i,i+batch_size * slide,slide)]),
                    np.array([data[j+window_size+1:j+window_size+1+out_size] for j in range(i,i+batch_size * slide,slide)])
                )
            except:
                break

def data_iterator_s2s(data,timestamps,window_size = 48,slide =8,out_size=8):
    #月・日・曜日・時間を[0,1]で正規化
    standard_time = np.array(
        [
            (
                date.month/12,
                date.day/31,
                date.weekday()/7,
                (date.hour+(date.minute/60))/24
            )
            for date in timestamps
        ]
    )
    return (
        [
            np.array([data[j:j+window_size] for j in range(0,len(data) - window_size - out_size,slide)]),
            np.array([standard_time[j+window_size] for j in range(0,len(data) - window_size - out_size,slide)]),
            np.array([data[j+window_size:j+window_size+1] for j in range(0,len(data) - window_size - out_size,slide)]),

        ],
        [
            np.array([data[j+window_size+1:j+window_size+1+out_size] for j in range(0,len(data) - window_size - out_size,slide)])
        ]
    )


def data_iterator(data,timestamps,window_size = 48,slide =8,out_size=8):
    #月・日・曜日・時間を[0,1]で正規化
    standard_time = np.array(
        [
            (
                date.month/12,
                date.day/31,
                date.weekday()/7,
                (date.hour+(date.minute/60))/24
            )
            for date in timestamps
        ]
    )
    return (
        [
            np.array([data[j:j+window_size] for j in range(0,len(data) - window_size - out_size,slide)]),
            np.array([standard_time[j+window_size+1:j+window_size+1+out_size] for j in range(0,len(data) - window_size - out_size,slide)])
        ],
        np.array([data[j+window_size+1:j+window_size+1+out_size] for j in range(0,len(data) - window_size - out_size,slide)])
    )

def data_iterator_tl(data,window_size = 48,slide =8,out_size=8):
    return (
        np.array([data[j:j+window_size] for j in range(0,len(data) - window_size - out_size,slide)]),
        np.array([data[j+window_size+1:j+window_size+1+out_size] for j in range(0,len(data) - window_size - out_size,slide)])
    )
