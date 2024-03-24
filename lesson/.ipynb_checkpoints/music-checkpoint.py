from music21 import *
import music21

# 定义一个旋律
melody = [
    ('C4', 1), ('D4', 1), ('E4', 1), ('F4', 1),
    ('G4', 2), ('F4', 1), ('E4', 1), ('D4', 1),
    ('C4', 2), ('D4', 1), ('E4', 1), ('F4', 1),
    ('G4', 2), ('F4', 1), ('E4', 1), ('D4', 1),
    ('C4', 2)
]

# 创建一个音乐流
stream = stream.Stream()

# 将旋律添加到音乐流中
for note, duration in melody:
    n = music21.note.Note(note)
    n.duration.quarterLength = duration
    stream.append(n)

# 设置节拍和调性
stream.insert(0, meter.TimeSignature('4/4'))
stream.insert(0, key.KeySignature(0))

# 将音乐流写入MIDI文件
stream.write('midi', fp='sea_inspired_melody.mid')
