module Observer
    using Random
    using OrderedCollections
    using Distributions
    using StatsBase
    using DelimitedFiles
    export lattice_initialize,get_neighbor,game_process,strategy_learning,evolution_game,lattice_initialize_cluster
    L=100
    n = L*L
    function lattice_initialize(L,n,p_observer)
        n_c = round(Int, n * (1 - p_observer) * 0.5)
        n_d= round(Int, n * (1 - p_observer) * 0.5)
        n_o = round(Int, n * p_observer)
        all_coordinates = [(i, j) for i in 1:L for j in 1:L]
        observer_coordinates = sample(all_coordinates, n_o,replace=false)
        lattice = zeros(Int8,L,L)
        zeros_array = fill(0, n_c)
        ones_array = fill(1, n_d)
        strategy = vcat(zeros_array, ones_array)
        strategy = shuffle(strategy)
        k = 1
        for i in 1 : L
            for j in 1 : L
                if (i,j) in observer_coordinates
                    lattice[i, j] = 2
                else
                    lattice[i,j] = strategy[k]
                    k+=1
                end
            end
        end
        payoffs = zeros(Float64,L,L)

        observer_strategy_dict = OrderedDict{Tuple{Int, Int}, Vector{Int}}()

        for coord in observer_coordinates
            strategy_array = fill(-1, 5)
            observer_strategy_dict[coord] = strategy_array
        end
        return lattice,observer_coordinates,payoffs,observer_strategy_dict
    end

    function lattice_initialize_cluster(L, p_observer,n)
        n_o = Int(round(L * L * p_observer))
        n_c = round(Int, n * (1 - p_observer) * 0.5)
        n_d= round(Int, n * (1 - p_observer) * 0.5)
        println("n_o=$n_o,n_c=$n_c,n_d=$n_d")
        lattice = zeros(Int, L, L)
        L_cluster_radius = Int(sqrt(n_o) / 2)
        L_cluster_upper_bound = Int(Int(L / 2) - L_cluster_radius + 1)
        L_cluster_lower_bound = Int(Int(L / 2) + L_cluster_radius)
        println("L_cluster_radius=$L_cluster_radius,L_cluster_upper_bound=$L_cluster_upper_bound,L_cluster_lower_bound=$L_cluster_lower_bound")
        lattice[L_cluster_upper_bound:L_cluster_lower_bound, L_cluster_upper_bound:L_cluster_lower_bound] .= 2
        strategy = []
        strategy = append!(strategy, [0 for i in 1:n_c])
        strategy = append!(strategy, [1 for i in 1:n_d])
        strategy = shuffle(strategy)
        k = 1
        observer_coordinates = []
        for i in 1:L
            for j in 1:L
                if lattice[i, j] == 2
                    push!(observer_coordinates,(i,j))
                else
                    lattice[i, j] = strategy[k]
                    k += 1
                end
            end
        end
  
        observer_strategy_dict = OrderedDict{Tuple{Int, Int}, Vector{Int}}()

        for coord in observer_coordinates
            coord = Tuple(coord)
            strategy_array = fill(-1, 5)
            observer_strategy_dict[coord] = strategy_array
        end
        payoffs = zeros(Int,L,L)
        return lattice,observer_coordinates,payoffs,observer_strategy_dict
    
    end


    function get_neighbor(a) 
        if a[1] == 1 || a[1] == L || a[2] == 1 || a[2] == L
            if a[1] == 1 && a[2] == 1
                return [[L,a[2]],[a[1]+1,a[2]],[a[1],L],[a[1],a[2]+1]]  
            elseif a[1] == 1 && a[2] == L
                return [[L,a[2]],[a[1]+1,a[2]],[a[1],a[2]-1],[a[1],1]]
            elseif a[1] == L && a[2] == 1
                return [[a[1]-1,a[2]],[1,a[2]],[a[1],L],[a[1],a[2]+1]]
            elseif a[1] == L && a[2] == L
                return [[a[1]-1,a[2]],[1,a[2]],[a[1],a[2]-1],[a[1],1]]
            elseif a[1] == 1
                return [[L,a[2]],[a[1]+1,a[2]],[a[1],a[2]-1],[a[1],a[2]+1]]
            elseif a[1] == L
                return [[a[1]-1,a[2]],[1,a[2]],[a[1],a[2]-1],[a[1],a[2]+1]]
            elseif a[2] == 1
                return [[a[1]-1,a[2]],[a[1]+1,a[2]],[a[1],L],[a[1],a[2]+1]]
            elseif a[2] == L
                return [[a[1]-1,a[2]],[a[1]+1,a[2]],[a[1],a[2]-1],[a[1],1]]
            end
        else
            return [[a[1]-1,a[2]],[a[1]+1,a[2]],[a[1],a[2]-1],[a[1],a[2]+1]]
        end
            
    end     
    function game_process(lattice,observer_coordinates, center, ϕ, R, L,payoffs,κ1,observer_strategy_dict,a,b,σ²,invest)
        strategys = [lattice[i[1], i[2]] for i in get_neighbor(center)]
        push!(strategys, lattice[center[1], center[2]])
        positions = get_neighbor(center)
        push!(positions, center)

        n_c = length(findall(x -> x == 0, strategys))
        n_o = length(findall(x -> x == 2, strategys))
        indices = Int[]

        invests = 0
        for i in 1:length(positions)
            if lattice[positions[i]...] == 0
                invests += invest[positions[i]...]
            end
        end

        for (idx, value) in enumerate(strategys)
            if value == 2
                push!(indices, idx)
            end
        end
        
        for idx in indices
            coord = (positions[idx][1],positions[idx][2])
            observer_strategy_array = observer_strategy_dict[coord]
            observer_strategy_array[idx] = observer_strategy_update(n_c,n_o,κ1,ϕ,invests,b)
            strategys[idx] = observer_strategy_array[idx]
            observer_strategy_dict[coord] = observer_strategy_array
        end
        n_c_l = length(findall(x -> x == 0, strategys))
        n_c_l = n_c_l - n_c 
        totle_invest = n_c*a + n_c_l*b
        if totle_invest >=ϕ
            π_c = (totle_invest * R / 5)
            π_d = totle_invest * R / 5
            π_l_c = (totle_invest * R / 5) - b
            for i in 1:length(positions)
                if lattice[positions[i][1],positions[i][2]]==0
                    payoffs[positions[i][1],positions[i][2]] += π_c - invest[positions[i][1],positions[i][2]]
                elseif lattice[positions[i][1],positions[i][2]]==1
                    payoffs[positions[i][1],positions[i][2]] += π_d
                elseif lattice[positions[i][1],positions[i][2]] == 2 &&strategys[i]==0
                    payoffs[positions[i][1],positions[i][2]] += π_l_c
                elseif lattice[positions[i][1],positions[i][2]] == 2 &&strategys[i]==1
                    payoffs[positions[i][1],positions[i][2]] += π_d
                end
            end
        end
        return lattice,payoffs,observer_strategy_dict

    end


    function observer_strategy_update(n_c,n_o,κ1,ϕ,invests,b)
        α = invests+n_o*b
            if invests >= ϕ || α < ϕ 
               return 1
            elseif invests < ϕ && α > ϕ
                    P =  ℯ ^ (-((n_o*b-ϕ+invests)/(ϕ-invests))/κ1)
                    if rand() < P
                        return 0
                    else
                        return 1
                    end            
            elseif invests < ϕ && α == ϕ  
                return 0
            else
            end
    end

    
    function strategy_learning(lattice,observer_coordinates,payoffs,κ,observer_strategy_dict)
        strategy_to_be = zeros(Int8,L,L)
        new_observer_strategy_dict = copy(observer_strategy_dict)
        for i in 1:L
            for j in 1:L
                game_neighbor=get_game_neighbor([i,j])
                chosen_neighbor = shuffle(game_neighbor)[1]
                f_i = payoffs[i,j]
                f_j = payoffs[chosen_neighbor[1],chosen_neighbor[2]]

                temp = 1 / (1 + ℯ ^ -((f_j - f_i)/κ))
                
                if rand() < temp
                    if lattice[i,j]!=2 && lattice[chosen_neighbor[1],chosen_neighbor[2]]==2
                        strategy_to_be[i,j] = 2
                        push!(observer_coordinates,(i,j))
                        strategy_array = fill(-1, 5)
                        new_observer_strategy_dict[(i, j)] = strategy_array
                    elseif lattice[i,j]!=2 && lattice[chosen_neighbor[1],chosen_neighbor[2]]!=2
                        strategy_to_be[i,j] = lattice[chosen_neighbor[1],chosen_neighbor[2]]
                    elseif lattice[i,j] == 2 && lattice[chosen_neighbor[1],chosen_neighbor[2]] == 2
                        strategy_to_be[i,j] = 2
                    elseif lattice[i,j] == 2 && lattice[chosen_neighbor[1],chosen_neighbor[2]] != 2
                        strategy_to_be[i,j] = lattice[chosen_neighbor[1],chosen_neighbor[2]]
                        filter!(x -> x != (i, j), observer_coordinates)
                        delete!(new_observer_strategy_dict, (i, j))
                    end
                else
                    strategy_to_be[i,j] = lattice[i,j]
                end
            end
        end
        return strategy_to_be,observer_coordinates,new_observer_strategy_dict
    end

    function evolution_game(time_step, lattice, R, temp,temp1, temp2,temp3,ϕ, κ1, payoffs, κ, observer_coordinates,p_observer,observer_strategy_dict,a,b,σ²)

        println("\nR=$R,ϕ=$ϕ,a=$a,b=$b,κ=$κ,p_observer=$p_observer")
        for t in 1:time_step
            payoffs = zeros(Float64,L,L)
            invest = zeros(Float64,L,L)
            
            for i in 1:L
                for j in 1:L
                    invest[i,j] = a
                end
            end
            for i in 1:L
                for j in 1:L
                    center = [i,j]
                    lattice,payoffs,observer_strategy_dict = game_process(lattice,observer_coordinates, center, ϕ, R, L,payoffs,κ1,observer_strategy_dict,a,b,σ²,invest)
                end
            end
    
            count = 0
            for strategy_array in values(observer_strategy_dict)
                count += length(findall(x -> x == 0, strategy_array))
            end
            n_observer = (length(findall(x->x==2,lattice)))
            push!(temp2,(length(findall(x->x==0,lattice))*5+count)/(L*L*5))
            push!(temp3,(length(findall(x->x==2,lattice)))/(L*L))
            
            push!(temp1,count/((L*L)*5))
            push!(temp,(length(findall(x->x==0,lattice)))/((L*L)))
            
            lattice_data = Matrix{Any}(undef, L,L)
            for i in 1:L
                for j in 1:L
                    if lattice[i,j] == 2
                        lattice_data[i,j] = observer_strategy_dict[(i,j)]
                    else
                        lattice_data[i,j] = lattice[i,j]
                    end
                end
            end
            strategy_to_be,observer_coordinates,observer_strategy_dict = strategy_learning(lattice,observer_coordinates,payoffs,κ,observer_strategy_dict)
            lattice = strategy_to_be
        end
        return lattice,temp,temp1,temp2,temp3
    end

    function get_game_neighbor(center)
       game_neighbors = get_neighbor(center)
       new_game_neighbors = copy(game_neighbors)
       for game_neighbor in game_neighbors
            game_neighbors_coordinates = get_neighbor(game_neighbor)
            for game_neighbors_coordinate in game_neighbors_coordinates
                if !(game_neighbors_coordinate in new_game_neighbors) && game_neighbors_coordinate != center
                    push!(new_game_neighbors,game_neighbors_coordinate)
                end
            end
        end
        return new_game_neighbors
    end
end


