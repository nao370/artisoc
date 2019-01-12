function is_bend() as Boolean{
	dim ret as Boolean

	if Rnd() < 0.01 then
		ret = true
	else
		ret = false
	end if
	
	return ret
}

function determining_bend() as Double{
	dim ret as Double

	ret = NormInv(Rnd(),0,30) + my.Direction
	
	return ret
}