
sub determining_vector(){
	
	calculate_vector()
	select_vector()
	
}

sub calculate_vector(){
//my.next_vectorに3種類計算する．2種類しかないときは空いてるところに(0,0,0)を入れる
	dim b_type as Integer
	dim angle as Double

	b_type = get_type_boundary()
	angle = calculate_angle(b_type)

	store_vector(b_type,angle)
}

sub store_vector(type as Integer,angle as Double){
//ベクトルをそれぞれ計算して格納する．この時に行けないベクトルも削除する
	dim R(3,3) as Double
	dim x(3) as Double
	dim u(3) as Integer
	dim length as Double

	type = my.b_type
	angle = my.angle_surface

	if type > 2 then
	//±の障害物見てベクトルの方向を決める
		if type == 3 then
			if universe.map.is_void(my.X,my.Y+1,my.Z) then
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = -1
				my.next_vector(0,2) = 0
			else
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = 1
				my.next_vector(0,2) = 0
			end if
			if universe.map.is_void(my.X+1,my.Y,my.Z) then
				my.next_vector(0,0) = -1
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = 0
			else
				my.next_vector(0,0) = 1
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = 0
			end if
		elseif type == 4 then
			if universe.map.is_void(my.X,my.Y,my.Z+1) then
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = -1
			else
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = 1
			end if
			if universe.map.is_void(my.X+1,my.Y,my.Z) then
				my.next_vector(0,0) = -1
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = 0
			else
				my.next_vector(0,0) = 1
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = 0
			end if
		elseif type == 5 then
			if universe.map.is_void(my.X,my.Y+1,my.Z) then
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = -1
				my.next_vector(0,2) = 0
			else
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = 1
				my.next_vector(0,2) = 0
			end if
			if universe.map.is_void(my.X,my.Y,my.Z+1) then
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = -1
			else
				my.next_vector(0,0) = 0
				my.next_vector(0,1) = 0
				my.next_vector(0,2) = 1
			end if
		else
			println("type miss")
		end if
	else
		if type == 0 then
			my.forward_vector(0,0) = 1
			my.forward_vector(0,1) = 0
			my.forward_vector(0,2) = 0

			my.forward_vector(1,0) = -1
			my.forward_vector(1,1) = 0
			my.forward_vector(1,2) = 0
			
			u(0) = 1
			u(1) = 0
			u(2) = 0

			calculate_matrix()

			x(0) = my.matrix(0,0) * my.forward_vector(0) + my.matrix(0,1) * my.forward_vector(1) + my.matrix(0,2) * my.forward_vector(2)
			x(1) = my.matrix(1,0) * my.forward_vector(0) + my.matrix(1,1) * my.forward_vector(1) + my.matrix(1,2) * my.forward_vector(2)
			x(2) = my.matrix(2,0) * my.forward_vector(0) + my.matrix(2,1) * my.forward_vector(1) + my.matrix(2,2) * my.forward_vector(2)

			length = Sqr( x(0)^2 + x(1)^2 + x(2)^2 )
			my.forward_vector(2,0) = x(0) / length
			my.forward_vector(2,1) = x(1) / length
			my.forward_vector(2,2) = x(2) / length
		elseif type == 1 then
			my.forward_vector(0,0) = 0
			my.forward_vector(0,1) = 1
			my.forward_vector(0,2) = 0

			my.forward_vector(1,0) = 0
			my.forward_vector(1,1) = -1
			my.forward_vector(1,2) = 0
			
			u(0) = 0
			u(1) = 1
			u(2) = 0

			calculate_matrix()

			x(0) = my.matrix(0,0) * my.forward_vector(0) + my.matrix(0,1) * my.forward_vector(1) + my.matrix(0,2) * my.forward_vector(2)
			x(1) = my.matrix(1,0) * my.forward_vector(0) + my.matrix(1,1) * my.forward_vector(1) + my.matrix(1,2) * my.forward_vector(2)
			x(2) = my.matrix(2,0) * my.forward_vector(0) + my.matrix(2,1) * my.forward_vector(1) + my.matrix(2,2) * my.forward_vector(2)

			length = Sqr( x(0)^2 + x(1)^2 + x(2)^2 )
			my.forward_vector(2,0) = x(0) / length
			my.forward_vector(2,1) = x(1) / length
			my.forward_vector(2,2) = x(2) / length
		elseif type == 2 then
			my.forward_vector(0,0) = 0
			my.forward_vector(0,1) = 0
			my.forward_vector(0,2) = 1

			my.forward_vector(1,0) = 0
			my.forward_vector(1,1) = 0
			my.forward_vector(1,2) = -1
			
			u(0) = 0
			u(1) = 0
			u(2) = 1

			calculate_matrix()

			x(0) = my.matrix(0,0) * my.forward_vector(0) + my.matrix(0,1) * my.forward_vector(1) + my.matrix(0,2) * my.forward_vector(2)
			x(1) = my.matrix(1,0) * my.forward_vector(0) + my.matrix(1,1) * my.forward_vector(1) + my.matrix(1,2) * my.forward_vector(2)
			x(2) = my.matrix(2,0) * my.forward_vector(0) + my.matrix(2,1) * my.forward_vector(1) + my.matrix(2,2) * my.forward_vector(2)

			length = Sqr( x(0)^2 + x(1)^2 + x(2)^2 )
			my.forward_vector(2,0) = x(0) / length
			my.forward_vector(2,1) = x(1) / length
			my.forward_vector(2,2) = x(2) / length		
		else
			PrintLn("type miss")
		end if
	end if
}

sub calculate_matrix(){
	dim theta as Double
	dim n as Double

	theta = DegreeToRad(my.angle_surface)
	if my.b_type == 0 then
		n(0) = 1
		n(1) = 0
		n(2) = 0
	elseif my.b_type == 1 then
		n(0) = 1
		n(1) = 0
		n(2) = 0	
	else
		n(0) = 1
		n(1) = 0
		n(2) = 0	
	end if

	my.matrix(0,0) = Cos(theta) + (n(0)^2) * (1 - Cos(theta))
	my.matrix(0,1) = n(0) * n(1) * (1 - Cos(theta)) - (n(2)) * (sin(theta))
	my.matrix(0,2) = n(2) * n(0) * (1 - Cos(theta)) + (n(1)) * (sin(theta))
	my.matrix(1,0) = n(0) * n(1) * (1 - Cos(theta)) + (n(2)) * (sin(theta))
	my.matrix(1,1) = Cos(theta) + (n(1)^2) * (1 - Cos(theta))
	my.matrix(1,2) = n(1) * n(2) * (1 - Cos(theta)) - (n(0)) * (sin(theta))
	my.matrix(2,0) = n(2) * n(0) * (1 - Cos(theta)) - (n(1)) * (sin(theta))
	my.matrix(2,1) = n(1) * n(2) * (1 - Cos(theta)) + (n(0)) * (sin(theta))
	my.matrix(2,2) = Cos(theta) + (n(2)^2) * (1 - Cos(theta))
}


function calculate_angle(type as Integer) as Double{
//my.forward_vectorの境界線に垂直な成分と壁のベクトルのなす角を計算して角度を返す
	//いらんかったかも
	return 0 
}

function get_type_boundary() as Integer{
//境界線がX軸に平行ならば0 Y軸に平行なら1 Z軸に平行なら2 x,y軸がそのてんからあるなら3 x,zがるなら4,yzがあるなら5を返す
//いらんかったかも
	return 0
}