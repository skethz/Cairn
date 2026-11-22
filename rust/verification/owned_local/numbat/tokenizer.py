"""Independent bounded UTF-8 byte BPE; data-only JSON vocabulary."""
from collections import Counter
import json
from pathlib import Path
from .local_io import read_local_bytes


class ByteBPETokenizer:
    pad_id,bos_id,eos_id=0,1,2
    max_text_bytes=1_000_000

    def __init__(self,merges=()):
        self.merges=[];self.pieces={i+3:bytes([i]) for i in range(256)}
        seen=set();size=256
        for pair in merges:
            if len(self.merges)>=131072-259 or type(pair) not in (tuple,list) or len(pair)!=2:
                raise ValueError('Invalid merge list')
            left,right=pair
            if type(left) is not int or type(right) is not int or left not in self.pieces or right not in self.pieces:
                raise ValueError('Invalid/future merge token')
            if (left,right) in seen: raise ValueError('Duplicate merge')
            content=self.pieces[left]+self.pieces[right];size+=len(content)
            if len(content)>65536 or size>64*1024*1024: raise ValueError('Vocabulary exceeds byte budget')
            self.pieces[259+len(self.merges)]=content;self.merges.append((left,right));seen.add((left,right))

    @property
    def vocabulary_size(self): return 259+len(self.merges)

    @staticmethod
    def _replace(sequence,pair,new_id):
        output=[];index=0
        while index<len(sequence):
            if index+1<len(sequence) and (sequence[index],sequence[index+1])==pair:
                output.append(new_id);index+=2
            else: output.append(sequence[index]);index+=1
        return output

    def encode(self,text,*,bos=False,eos=False):
        if type(text) is not str or len(text)>self.max_text_bytes: raise ValueError('Text exceeds budget')
        encoded=text.encode('utf-8')
        if len(encoded)>self.max_text_bytes or len(encoded)*max(1,len(self.merges))>20_000_000:
            raise ValueError('Tokenization work exceeds budget')
        tokens=[byte+3 for byte in encoded]
        for rank,pair in enumerate(self.merges): tokens=self._replace(tokens,pair,259+rank)
        return ([self.bos_id] if bos else [])+tokens+([self.eos_id] if eos else [])

    def decode(self,ids,*,skip_special=True,errors='strict'):
        if errors not in ('strict','replace','ignore'): raise ValueError('Unsupported decoding policy')
        pieces=[];size=0
        for index,token in enumerate(ids):
            if index>=self.max_text_bytes or type(token) is not int or not 0<=token<self.vocabulary_size:
                raise ValueError('Invalid token or oversized sequence')
            if token<3:
                if skip_special: continue
                piece=(b'<pad>',b'<bos>',b'<eos>')[token]
            else: piece=self.pieces[token]
            size+=len(piece)
            if size>self.max_text_bytes: raise ValueError('Decoded text exceeds budget')
            pieces.append(piece)
        return b''.join(pieces).decode('utf-8',errors=errors)

    @classmethod
    def train(cls,texts,*,vocabulary_size=512,minimum_frequency=2):
        if type(vocabulary_size) is not int or not 259<=vocabulary_size<=131072: raise ValueError('Invalid vocabulary size')
        if type(minimum_frequency) is not int or minimum_frequency<1: raise ValueError('Invalid frequency')
        documents=[];total=0
        for index,text in enumerate(texts):
            if index>=10000 or type(text) is not str or len(text)>cls.max_text_bytes: raise ValueError('Corpus exceeds budget')
            encoded=text.encode('utf-8');total+=len(encoded)
            if total>cls.max_text_bytes or total*max(1,vocabulary_size-259)>20_000_000: raise ValueError('Training work exceeds budget')
            documents.append([byte+3 for byte in encoded])
        merges=[]
        while 259+len(merges)<vocabulary_size:
            counts=Counter(pair for seq in documents for pair in zip(seq,seq[1:]))
            if not counts: break
            pair=min(counts,key=lambda p:(-counts[p],p))
            if counts[pair]<minimum_frequency: break
            token=259+len(merges)
            documents=[cls._replace(seq,pair,token) for seq in documents];merges.append(pair)
        return cls(merges)

    def save(self,path):
        destination=Path(path)
        if destination.is_symlink() or '://' in str(path): raise ValueError('Expected a local destination')
        destination.write_text(json.dumps({'format':'numbat-byte-bpe','version':1,'merges':self.merges})+'\n')

    @classmethod
    def load(cls,path):
        data=json.loads(read_local_bytes(path,max_bytes=8_000_000))
        if type(data) is not dict or set(data)!={'format','version','merges'} or data['format']!='numbat-byte-bpe' or data['version']!=1 or type(data['merges']) is not list:
            raise ValueError('Unsupported vocabulary format')
        return cls(data['merges'])
