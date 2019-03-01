# Alphabet-Inc.-GOOGL-Stock-Prices-using-sklearn

It is simple implementation using sklearn to predict stock price for Alphabet Inc. GOOGL Stock Prices taken from Quandl.
It deals with use of preprocessing, cross-validation and models from sklearn. This small project is learnt from Harrison Kinsley's tutorial. It covers some of the speacial features of python libraries and modules like unix timestamp.

Note:

posix is the set of description of operating system ...specially UNIX based OS..
posix time or unix time or unix epoch time or unix timestamp is the time elapsed in seconds after
1st JAN 1970(thursday) and is set as UTC.
unix time system is designed in C under integer data type whose value is upto
2147483647 i.e. when 2147483647 seconds or 68 years are elapsed then seconds overflow
due to int datatype and this event will be take place on  19/jan/2038.

Why adj close is taken as feature for prediction? Why not the others features?
=>
It is because we always choose the feature that is sustaining direct impact. HL_per itself
is calulated from adj close price so it is not having that direct impact. Also,if we come to
predict HL_per then the prediction should lie only in the range: 0 to 100,which will make
plots more closer to each other and somewhat linear with x-axis,we wont be finding
the variation that we want to make our model worth satisfying with high accuracy.
