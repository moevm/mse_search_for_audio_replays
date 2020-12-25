#!/usr/bin/env python

from repetitions import get_repetitions
from audio import load_audio, resample_to_common
from progress import simple_progressbar
import os

def comparison(output, answer):
	res = open(output, "r")
	out = res.readlines()
	res.close()
	req = open(answer, "r")
	ans = req.readlines()
	req.close()
	if len(ans) > len(out):
		return 0
	for i in range(len(ans)):
		newLineAns = ans[i].split(' ')
		#print(newLineAns)
		
		timeBeginAnsFirst = newLineAns[2].split('--')[0]
		timeBeginAnsFirstHours = newLineAns[2].split('--')[0].split(':')[0]
		timeBeginAnsFirstMinuts = newLineAns[2].split('--')[0].split(':')[1]
		timeBeginAnsFirstSeconds = newLineAns[2].split('--')[0].split(':')[2]
		timeEndAnsFirst = newLineAns[2].split('--')[1]
		timeEndAnsFirstHours = newLineAns[2].split('--')[1].split(':')[0]
		timeEndAnsFirstMinuts = newLineAns[2].split('--')[1].split(':')[1]
		timeEndAnsFirstSeconds = newLineAns[2].split('--')[1].split(':')[2]
		
		
		timeBeginAnsSecond = newLineAns[5].split('--')[0]
		timeBeginAnsSecondHours = newLineAns[5].split('--')[0].split(':')[0]
		timeBeginAnsSecondMinuts = newLineAns[5].split('--')[0].split(':')[1]
		timeBeginAnsSecondSeconds = newLineAns[5].split('--')[0].split(':')[2]
		timeEndAnsSecond = newLineAns[5].split('--')[1]
		timeEndAnsSecondHours = newLineAns[5].split('--')[1].split(':')[0]
		timeEndAnsSecondMinuts = newLineAns[5].split('--')[1].split(':')[1]
		timeEndAnsSecondSeconds = newLineAns[5].split('--')[1].split(':')[2]
		
		
		for j in range(len(out)):
			newLineOut = out[j].split(' ')
			#print(newLineOut)
			if newLineOut[1] == newLineAns[1] and newLineOut[4] == newLineAns[4]:
				k = 0
				timeBeginOutFirst = newLineOut[2].split('--')[0]
				timeBeginOutFirstHours = newLineOut[2].split('--')[0].split(':')[0]
				timeBeginOutFirstMinuts = newLineOut[2].split('--')[0].split(':')[1]
				timeBeginOutFirstSeconds = newLineOut[2].split('--')[0].split(':')[2].split('.')[0]
				timeEndOutFirst = newLineOut[2].split('--')[1]
				timeEndOutFirstHours = newLineOut[2].split('--')[1].split(':')[0]
				timeEndOutFirstMinuts = newLineOut[2].split('--')[1].split(':')[1]
				timeEndOutFirstSeconds = newLineOut[2].split('--')[1].split(':')[2].split('.')[0]
				
				timeBeginOutSecond = newLineOut[5].split('--')[0]
				timeBeginOutSecondHours = newLineOut[5].split('--')[0].split(':')[0]
				timeBeginOutSecondMinuts = newLineOut[5].split('--')[0].split(':')[1]
				timeBeginOutSecondSeconds = newLineOut[5].split('--')[0].split(':')[2].split('.')[0]
				timeEndOutSecond = newLineOut[5].split('--')[1]
				timeEndOutSecondHours = newLineOut[5].split('--')[1].split(':')[0]
				timeEndOutSecondMinuts = newLineOut[5].split('--')[1].split(':')[1]
				timeEndOutSecondSeconds = newLineOut[5].split('--')[1].split(':')[2].split('.')[0]
				if timeBeginAnsFirst == timeBeginOutFirst and timeEndAnsFirst == timeEndOutFirst and timeBeginAnsSecond == timeBeginOutSecond and timeEndAnsSecond == timeEndOutSecond:
					return 1
				if timeBeginAnsFirstHours == timeBeginOutFirstHours and timeBeginAnsFirstMinuts == timeBeginOutFirstMinuts and abs(int(timeBeginAnsFirstSeconds) - int(timeBeginOutFirstSeconds)) < 3:
					k = k + 1
				if timeEndAnsFirstHours == timeEndOutFirstHours and timeEndAnsFirstMinuts == timeEndOutFirstMinuts and abs(int(timeEndAnsFirstSeconds) - int(timeEndOutFirstSeconds)) < 3:
					k = k + 1
				if timeBeginAnsSecondHours == timeBeginOutSecondHours and timeBeginAnsSecondMinuts == timeBeginOutSecondMinuts and abs(int(timeBeginAnsSecondSeconds) - int(timeBeginOutSecondSeconds)) < 3:
					k = k + 1
				if timeEndAnsSecondHours == timeEndOutSecondHours and timeEndAnsSecondMinuts == timeEndOutSecondMinuts and abs(int(timeEndAnsSecondSeconds) - int(timeEndOutSecondSeconds)) < 3:
					k = k + 1
				if k == 4:
					return 1
			else:
				if newLineOut[1] == newLineAns[4] and newLineOut[4] == newLineAns[1]:
					cell1 = newLineOut[4]
					cell2 = newLineOut[5]
					cell4 = newLineOut[1]
					cell5 = newLineOut[2]
					newLineOut[1] = cell1
					newLineOut[2] = cell2
					newLineOut[4] = cell4
					newLineOut[5] = cell5
					k = 0
					timeBeginOutFirst = newLineOut[2].split('--')[0]
					timeBeginOutFirstHours = newLineOut[2].split('--')[0].split('.')[0]
					timeBeginOutFirstMinuts = newLineOut[2].split('--')[0].split('.')[1]
					timeBeginOutFirstSeconds = newLineOut[2].split('--')[0].split('.')[2]
					timeEndOutFirst = newLineOut[2].split('--')[1]
					timeEndOutFirstHours = newLineOut[2].split('--')[1].split('.')[0]
					timeEndOutFirstMinuts = newLineOut[2].split('--')[1].split('.')[1]
					timeEndOutFirstSeconds = newLineOut[2].split('--')[1].split('.')[2]
					
					timeBeginOutSecond = newLineOut[5].split('--')[0]
					timeBeginOutSecondHours = newLineOut[5].split('--')[0].split('.')[0]
					timeBeginOutSecondMinuts = newLineOut[5].split('--')[0].split('.')[1]
					timeBeginOutSecondSeconds = newLineOut[5].split('--')[0].split('.')[2]
					timeEndOutSecond = newLineOut[5].split('--')[1]
					timeEndOutSecondHours = newLineOut[5].split('--')[1].split('.')[0]
					timeEndOutSecondMinuts = newLineOut[5].split('--')[1].split('.')[1]
					timeEndOutSecondSeconds = newLineOut[5].split('--')[1].split('.')[2]
					if timeBeginAnsFirst == timeBeginOutFirst and timeEndAnsFirst == timeEndOutFirst and timeBeginAnsSecond == timeBeginOutSecond and timeEndAnsSecond == timeEndOutSecond:
						return 1
					if timeBeginAnsFirstHours == timeBeginOutFirstHours and timeBeginAnsFirstMinuts == timeBeginOutFirstMinuts and abs(int(timeBeginAnsFirstSeconds) - int(timeBeginOutFirstSeconds)) < 3:
						k = k + 1
					if timeEndAnsFirstHours == timeEndOutFirstHours and timeEndAnsFirstMinuts == timeEndOutFirstMinuts and abs(int(timeEndAnsFirstSeconds) - int(timeEndOutFirstSeconds)) < 3:
						k = k + 1
					if timeBeginAnsSecondHours == timeBeginOutSecondHours and timeBeginAnsSecondMinuts == timeBeginOutSecondMinuts and abs(int(timeBeginAnsSecondSeconds) - int(timeBeginOutSecondSeconds)) < 3:
						k = k + 1
					if timeEndAnsSecondHours == timeEndOutSecondHours and timeEndAnsSecondMinuts == timeEndOutSecondMinuts and abs(int(timeEndAnsSecondSeconds) - int(timeEndOutSecondSeconds)) < 3:
						k = k + 1
					if k == 4:
						return 1
					
		
	return 0
		

def start(fnames):
	def timestr(seconds_fp):
		mseconds = round(seconds_fp * 1e3)
		mseconds_only = mseconds % 1000
		seconds = mseconds // 1000
		seconds_only = seconds % 60
		minutes = seconds // 60
		minutes_only = minutes % 60
		hours = minutes // 60
		return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes_only, seconds_only, mseconds_only)
	signals, rate = resample_to_common(
        load_audio(fname, normalize=True)
        for fname in fnames
    )
	with simple_progressbar('Detecting repetitions') as bar, open('output.txt', 'w') as sourceFile:
		for t1, t2, l, p in get_repetitions(signals, rate, progress=bar.update):
			i1, tt1 = t1
			i2, tt2 = t2
			percent = 100 * p
			print("repetition: {} {}--{} <=> {} {}--{} ({:.1f}%)".format(fnames[i1], timestr(tt1), timestr(tt1+l),fnames[i2], timestr(tt2), timestr(tt2+l),percent), file = sourceFile)


#print(os.getcwd())
os.chdir('../')
#print(os.getcwd())
dirs = os.listdir();
try:
	index = dirs.index('tests')
except:
	index = -1

if index == -1:
	print('Каталог с теставыми записями не найден, поместите его в корень проекта.')
	exit(0)

os.chdir(os.getcwd() + '/tests')
#print(os.getcwd())
#print(os.listdir())
tests = os.listdir() #Папки с тестами
tests.sort()
for test in tests:
	path = os.getcwd()
	if os.path.isdir(path + '/' + test):
		os.chdir(path + '/' + test)
		l = os.listdir();
		files = []
		for i in l:
			try:
				st = i.split('.')
				if st[1] == 'mp3':
					print(i)
					files.append(i)
				#else:
					#print("{} NO .mp3".format(i))
			except:
				print("{} NO .mp3".format(i))
		print(files)
		if len(files) > 0:
			try:
				start(files)
			except Exception as e:
				print("{} : {}".format(type(e).__name__, e))
			verdict = comparison(os.getcwd()+"/"+'output.txt', os.getcwd()+"/"+'answer.txt')
			if verdict:
				print("TEST PASSED")
			else:
				print("TEST FAILED")
		os.chdir(path)
	
