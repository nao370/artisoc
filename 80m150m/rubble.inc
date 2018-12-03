sub make_N_rubble(N as Integer, rate as double){
	dim i as integer
	dim j as integer
	dim set as AgtSet
	dim a as Integer
	dim b as Integer
	dim one as agt
	dim temp as AgtSet

	for j = 1 to GetHeightSpace(universe.map) - (N + 1)
		for i = 1 to GetWidthSpace(universe.map) - (N + 1)
	  	if rnd()*100 < rate and (i > 10 or j > 10) then
	  		MakeOneAgtSetAroundPosition(temp,universe.map,i,j,0,0,universe.map.barrier)
	  		if CountAgtSet(temp) == 0 then
	  			set = CreateAgtMulti(universe.map.barrier,(N^2))
	  			for each one in set
	  				a = GetAgtEntry(set,one) / N 
	  				b = GetAgtEntry(set,one) Mod N
	  				one.X = i + a
	  				one.Y = j + b
	  			next one
	    	end if
	    end if
	   next i
	  next j

}