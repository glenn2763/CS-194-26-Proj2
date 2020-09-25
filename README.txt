To sharpen an image:

open main.py and go to lines 128-132 and uncomment whichever image you wish to sharpen, then 
go to lines 135-138 and make sure line 136 (sharpen('data/' + imname)) is the only line of those uncommented.
Finally, from the base project directory, run 'python main.py'.

To get edges of an image:

open main.py and go to lines 128-132 and uncomment whichever image you wish to get edges of, then 
go to lines 135-138 and make sure line 135 (get_edges('data/' + imname)) is the only line of those uncommented.
*The threshold for the code is set only for cameraman.jpg and thus doesn't look good on other images*
Finally, from the base project directory, run 'python main.py'.

To straighten an image:

run 'python rotate.py' from the base project directory
*because the facade file is so large, this takes about 45 seconds to run so you gotta be patient*

To make a hybrid image:

run 'python hybrid.py' from the base project directory.

To blend the apple and orange:

run 'python blend.py' from the base project directory.
