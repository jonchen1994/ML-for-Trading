Contents:

indicators.py - Contains functions for calculating 5 technical indicators, all require price as a vector for input:
	SMA(prices, window) - returns single vector
	EMA(prices, span) - returns single vector
	MACD(prices, quick = 26, slow = 12, signal = 9) - returns 2 column DF
	momentum(prices, window) - returns single vector
	bollingerBand(prices, window=20, width = 2) - returns single signal vector
	
	indicators should be imported and fed prices. Functions do not inherently 		normalize so that must be done prior to feeding into functions if you want 		normalized output. 
	
TheoreticallyOptimalStrategy.py - Contains the function for a single column DF:
	testPolicy(symbol="JPM",sd = dt.datetime(2008, 1, 1),
    ed = dt.datetime(2009, 12, 31),
    sv =100000,
    commission=0.0,
    impact=0.00)
    - returns single column DF indexed by date. 
    
marketsimcode.py - Contains functions for calculating portfolio value:
	compute_portvals(
        orders,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
	): - returns single column of portfolio values indexed by date, calls other functions along the steps. 

Input for orders must follow the format: index (by date), Symbol (string), Shares (number - positive means buy negative means sell). 

testproject.py - Run this file to test Optimal strategy vs benchmark, as well as create graphs describing indicator functions. 
