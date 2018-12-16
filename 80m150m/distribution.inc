function Distribution() as Double{
	dim v(2) as Double
  dim temp as AgtSet
  dim neighbor as AgtSet
  dim alignment as AgtSet
  dim d as Double
  dim far as AgtSet
  dim T as Double
  dim one as Agt
  dim count as Integer

  v(0) = 0
  v(1) = 0
	MakeOneAgtSetAroundOwn(neighbor,4,universe.map.robot,false)
	MakeOneAgtSetAroundOwn(far,10,universe.map.robot,false)
	DelAgtSet(far,alignment)
	DelAgtSet(alignment,neighbor)

	if CountAgtSet(neighbor) > 0 then //‹ß‚·‚¬‚éê‡‚ÌË—Í
	  for each one in neighbor
	    d = MeasureDistance(one.X,one.Y,my.X,my.Y,universe.map)/4
	    if d != 0 then
		  	v(0) = v(0) + (my.X - one.X)/(d^3)
		  	v(1) = v(1) + (my.Y - one.Y)/(d^3)
		  end if
	  next one
  end if

  MakeOneAgtSetAroundOwn(temp,1,universe.map.barrier,false)
  if CountAgtSet(temp) > 0 then //áŠQ•¨‚©‚ç‚ÌË—Í
	  for each one in temp
	    d = MeasureDistance(one.X,one.Y,my.X,my.Y,universe.map)
	    if d != 0 then
		  	v(0) = v(0) + 5 * (my.X - one.X)/(d^10) 
		  	v(1) = v(1) + 5 *  (my.Y - one.Y)/(d^10) 
		  end if
	  next one
  end if
  
  if CountAgtSet(far) > 0 then //‰“‚¢ŒÂ‘Ì‚©‚ç‚Ìˆø—Í
  	for each one in neighbor
  		d = MeasureDistance(one.X,one.Y,my.X,my.Y,universe.map)/5
		  v(0) = v(0) + (one.X - my.X) * (d ^ 2)
	  	v(1) = v(1) + (one.Y - my.Y) * (d ^ 2)
	  next one
  end if


	v(0) = v(0) + my.X
	v(1) = v(1) + my.Y

	return v
}