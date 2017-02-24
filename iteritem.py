#Dict.items() vs Dict.iteritems() 
#!/usr/bin/env python 
import datetime 

start = datetime.datetime.now()
d={1:'one',2:'two',3:'three'}
print 'd.items():'
for k,v in d.items():
   if d[k] is v: print '\tthey are the same object' 
   else: print '\tthey are different'
print "It takes", start-datetime.datetime.now(), "for d.items()"

start = datetime.datetime.now()
print 'd.iteritems():'   
for k,v in d.iteritems():
   if d[k] is v: print '\tthey are the same object' 
   else: print '\tthey are different'
print "It takes", start-datetime.datetime.now(), "for d.iteritems()"  
