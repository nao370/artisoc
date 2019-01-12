sub select_vector(){
//my.next_vector‚©‚ç1‚ÂŠm—¦‚Å‘I‚ñ‚Åmy.forward_vector‚É“ü‚ê‚é
	dim i as Integer
	dim prob(3) as Double
	dim r as Double
	dim sum_diff as Double
	
	make_base_vector()
	sum_diff = calclator_sum_diff()
	for i = 0 to 2
		prob(i) = calculate_probabilistic(my.next_vector(i,0),my.next_vector(i,1),my.next_vector(i,2),sum_diff)
	next i
	r = Rnd()

	if r < prob(0) then
		my.forward_vector(0) = my.next_vector(0,0)
		my.forward_vector(1) = my.next_vector(0,1)
		my.forward_vector(2) = my.next_vector(0,2)				
	elseif r < prob(0) + prob(1) then
		my.forward_vector(0) = my.next_vector(1,0)
		my.forward_vector(1) = my.next_vector(1,1)
		my.forward_vector(2) = my.next_vector(1,2)	
	else
		my.forward_vector(0) = my.next_vector(2,0)
		my.forward_vector(1) = my.next_vector(2,1)
		my.forward_vector(2) = my.next_vector(2,2)	
	end if
}

function calclator_sum_diff()as Double{
	dim diff as Double
	dim i as Integer

	diff = 0
	for i = 0 to 3
		diff = diff + Sqr( ( my.base_vector(0) - my.next_vector(i,0) )^2 + ( my.base_vector(1) - my.next_vector(i,1) )^2 + ( my.base_vector(2) - my.next_vector(i,2) )^2 )
	next i

	return diff
}

sub make_base_vector(){
	dim g(3) as Double
	dim a(3) as Double
	dim length as Double
	dim i as Integer

	g(0) = 0
	g(1) = 0
	g(2) = -1

	for i = 0 to 2
		a(i) = my.forward_vector(i) + g(i) * (49 / 60)
	next i

	length = Sqr( a(0)^2 + a(1) ^2 + a(2) ^ 2 )

	for i = 0 to 2
		my.base_vector(i) = a(i) / length
	next i
	
}

function calculate_probabilistic(x as Double,y as Double,z as Double, sum_diff as Double) as Double{
	dim diff as Double
	dim ret as Double
	dim n as Integer

	if my.next_vector(2,0) == 0 and my.next_vector(2,1) == 0 and my.next_vector(2,2) == 0 then
		n = 2
	else
		n = 3
	end if

	diff = Sqr( ( my.base_vector(0) - x )^2 + ( my.base_vector(1) - y )^2 + ( my.base_vector(2) - y )^2 )
	ret = ( 1 - diff / sum_diff ) / ( n -1 )
	
	return ret
}