
function different_surface() as Boolean{
	dim ret  as Boolean
	if is_appear_surface() or is_disappear_surface() then
		ret = true
	else
		ret = false
	end if
	
	return ret
}

function direction_type() as Integer{
//x平行0,xyで動く1,xzで動く2，y平行3，yzで動く4，z平行5の0~5をタイプで返す
	dim ret as Integer
	dim x as Double
	dim y as Double
	dim z as Double

	x = my.forward_vector(0)
	y = my.forward_vector(1)
	z = my.forward_vector(2)
	
	if x==0 then
		if y == 0 then
			ret = 5
		elseif z == 0 then
			ret = 3
		else
			ret = 4
		end if
	elseif y == 0 then
		if z == 0 then
			ret = 0
		else
			ret = 2
		end if	
	elseif z == 0 then
		ret = 1
	else
		PrintLn("miss")
	end if

	return ret
}

function is_appear_surface() as Boolean{
	dim ret as Boolean
	dim axis as Integer
	dim nx as Integer
	dim ny as Integer
	dim nz as Integer

	axis = direction_type()
	ny = Int(my.Y)
	nz = Int(my.Z)
	nx = Int(my.X)

	if axis == 0 then
		if nx == my.X then
			if my.forward_vector(0) > 0 then
				if universe.map.is_void(my.X+1,my.Y,my.Z) then
					ret = true
					my.angle_surface = -1
					my.b_type = 5
				else
					ret = false
				end if
			else
				if universe.map.is_void(my.X-1,my.Y,my.Z) then
					ret = true
					my.angle_surface = -1
					my.b_type = 5
				else
					ret = false
				end if			
			end if
		else
			ret = false
		end if
	elseif axis == 1 then
		if nx == my.X then
			if my.forward_vector(0) > 0 then
				if universe.map.is_void(nx+1,ny,nz) or universe.map.is_void(nx+1,ny+1,nz) then
					ret = true
					my.angle_surface = 90
					my.b_type = 1
				else
					ret = false
				end if
			else
				if universe.map.is_void(nx-1,ny,nz) or universe.map.is_void(nx-1,ny+1,nz) then
					ret = true
					my.angle_surface = 270
					my.b_type = 1
				else
					ret = false
				end if
			end if
		elseif ny == my.Y then
			if my.forward_vector(1) > 0 then
				if universe.map.is_void(nx,ny+1,nz) or universe.map.is_void(nx+1,ny,nz) then
					ret = true
					my.angle_surface = 270
					my.b_type = 0
				else
					ret = false
				end if
			else
				if universe.map.is_void(nx,ny-1,nz) or universe.map.is_void(nx+1,ny-1,nz) then
					ret = true
					my.angle_surface = 90
					my.b_type = 0
				else
					ret = false
				end if
			end if
		else
			ret = false
		end if
	elseif axis == 2 then	
		if nx == my.X then
			if my.forward_vector(0) > 0 then
				if universe.map.is_void(nx+1,ny,nz) or universe.map.is_void(nx+1,ny,nz+1) then
					ret = true
					my.angle_surface = 90
					my.b_type =2
				else
					ret = false
				end if
			else
				if universe.map.is_void(nx-1,ny,nz) or universe.map.is_void(nx-1,ny,nz+1) then
					ret = true
					my.angle_surface = 270
					my.b_type = 2
				else
					ret = false
				end if
			end if
		elseif nz == my.Z then
			if my.forward_vector(2) > 0 then
				if universe.map.is_void(nx,ny,nz+1) or universe.map.is_void(nx+1,ny,nz+1) then
					ret = true
					my.angle_surface = 270
					my.b_type = 0
				else
					ret = false
				end if
			else
				if universe.map.is_void(nx,ny,nz-1) or universe.map.is_void(nx+1,ny,nz-1) then
					ret = true
					my.angle_surface = 90
					my.b_type = 0
				else
					ret = false
				end if
			end if
		else
			ret = false
		end if	
	elseif axis == 3 then
		if ny == my.Y then
			if my.forward_vector(1) > 0 then
				if universe.map.is_void(my.X,my.Y+1,my.Z) then
					ret = true
					my.angle_surface = -1
					my.b_type =4
				else
					ret = false
				end if
			else
				if universe.map.is_void(my.X,my.Y-1,my.Z) then
					ret = true
					my.angle_surface = -1
					my.b_type =4
				else
					ret = false
				end if			
			end if
		else
			ret = false
		end if	
	elseif axis == 4 then
		if ny == my.y then
			if my.forward_vector(1) > 0 then
				if universe.map.is_void(nx,ny+1,nz) or universe.map.is_void(nx,ny+1,nz+1) then
					ret = true
					my.angle_surface = 90
					my.b_type =2
				else
					ret = false
				end if
			else
				if universe.map.is_void(nx,ny-1,nz) or universe.map.is_void(nx,ny-1,nz+1) then
					ret = true
					my.angle_surface = 270
					my.b_type =2
				else
					ret = false
				end if
			end if
		elseif nz == my.Z then
			if my.forward_vector(2) > 0 then
				if universe.map.is_void(nx,ny,nz+1) or universe.map.is_void(nx,ny+1,nz) then
					ret = true
					my.angle_surface = 270
					my.b_type =1
				else
					ret = false
				end if
			else
				if universe.map.is_void(nx,ny,nz-1) or universe.map.is_void(nx,ny+1,nz-1) then
					ret = true
					my.angle_surface = 90
					my.b_type =1
				else
					ret = false
				end if
			end if
		else
			ret = false
		end if		
	else
		if nz == my.Z then
			if my.forward_vector(1) > 0 then
				if universe.map.is_void(my.X,my.Y,my.Z+1) then
					ret = true
					my.angle_surface = -1
					my.b_type =3
				else
					ret = false
				end if
			else
				if universe.map.is_void(my.X,my.Y,my.Z-1) then
					ret = true
					my.angle_surface = -1
					my.b_type =3
				else
					ret = false
				end if			
			end if
		else
			ret = false
		end if		
	end if		

	return ret
}

function is_disappear_surface() as Boolean{
	dim ret as Boolean
	dim axis as Integer
	dim nx as Integer
	dim ny as Integer
	dim nz as Integer

	axis = direction_type()
	ny = Int(my.Y)
	nz = Int(my.Z)
	nx = Int(my.X)

	if axis == 0 then
		if nx == my.X then
			if not universe.map.is_void(nx,ny,nz-my.normal_vector(2)) then
				ret = true
				my.angle_surface = -1
				my.b_type = 5
			else
				if not universe.map.is_void(nx,ny-my.normal_vector(1),nz) then
					ret = true
					my.angle_surface = -1
					my.b_type = 5
				else
				 ret = false
				end if
			end if
		else
			ret = false
		end if
	elseif axis == 1 then
		if nx == my.X then
			if not universe.map.is_void(nx,ny,nz - my.normal_vector(2)) then
				ret = true
				if my.normal_vector(2) * my.forward_vector(0) > 0  then
					my.angle_surface = 270
				else
					my.angle_surface = 90
				end if
				my.b_type = 1
			else
				ret = false
			end if
		elseif ny == my.Y then
			if not universe.map.is_void(nx,ny,nz - my.normal_vector(2)) then
				ret = true
				if my.normal_vector(2) > 0 * my.forward_vector(1) > 0 then
					my.angle_surface = 270
				else
					my.angle_surface = 90
				end if
				my.b_type = 0
			else
				ret = false
			end if
		else
			ret = false
		end if
	elseif axis == 2 then
		if nx == my.X then
			if not universe.map.is_void(nx,ny - my.normal_vector(1),nz) then
				ret = true
				if my.normal_vector(1) * my.forward_vector(0) > 0  then
					my.angle_surface = 270
				else
					my.angle_surface = 90
				end if
				my.b_type = 2
			else
				ret = false
			end if
		elseif nz == my.Z then
			if not universe.map.is_void(nx,ny - my.normal_vector(1),nz) then
				ret = true
				if my.normal_vector(1) > 0 * my.forward_vector(2) > 0 then
					my.angle_surface = 270
				else
					my.angle_surface = 90
				end if
				my.b_type = 0
			else
				ret = false
			end if
		else
			ret = false
		end if	
	elseif axis == 3 then
		if nx == my.X then
			if not universe.map.is_void(nx,ny,nz-my.normal_vector(2)) then
				ret = true
				my.angle_surface = -1
				my.b_type = 5
			else
				if not universe.map.is_void(nx,ny-my.normal_vector(1),nz) then
					ret = true
					my.angle_surface = -1
					my.b_type = 5
				else
					ret = false
				end if
			end if
		else
			ret = false
		end if
	elseif axis == 4 then
		if ny == my.Y then
			if not universe.map.is_void(nx-my.normal_vector(0),ny,nz) then
				ret = true
				if my.normal_vector(0) * my.forward_vector(1) > 0  then
					my.angle_surface = 270
				else
					my.angle_surface = 90
				end if
				my.b_type = 2
			else
				ret = false
			end if
		elseif nz == my.Z then
			if not universe.map.is_void(nx - my.normal_vector(0),ny,nz) then
				ret = true
				if my.normal_vector(0) > 0 * my.forward_vector(2) > 0 then
					my.angle_surface = 270
				else
					my.angle_surface = 90
				end if
				my.b_type = 1
			else
				ret = false
			end if
		else
			ret = false
		end if	
	else
		if nx == my.X then
			if not universe.map.is_void(nx,ny,nz-my.normal_vector(2)) then
				ret = true
				my.angle_surface = -1
				my.b_type = 5
			else
				if not universe.map.is_void(nx,ny-my.normal_vector(1),nz) then
					ret = true
					my.angle_surface = -1
					my.b_type = 5
				else
					ret = false
				end if
			end if
		else
			ret = false
		end if
	end if		
	
	return ret
}