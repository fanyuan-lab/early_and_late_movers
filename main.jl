using Random
using DelimitedFiles
using StatsBase
include("observer_moudle.jl")
using .Observer

L=100
n = L*L
p_observers=[0,0.3,0.6,0.9,1.0]
ϕ = 3
R = 3.2
time_step = 100
a = 0.6
σ² = 0.2
b = 0.8
κ = 0.5
κ1 = 1

Threads.@threads for ave_time in 1:10
    cooprator_rates = Matrix{Any}(undef,time_step,length(p_observers)+1)
    observer_rates = Matrix{Any}(undef,time_step,length(p_observers)+1)
    all_cooprator_rates = Matrix{Any}(undef,time_step,length(p_observers)+1)
    Pobserver_rates = Matrix{Any}(undef,time_step,length(p_observers)+1)


    for (i,p_observer) in enumerate(p_observers)
        
        temp = []
        temp1 = []
        temp2 = []
        temp3 = []

        lattice,observer_coordinates,payoffs,observer_strategy_dict = lattice_initialize(L,n,p_observer)
        lattice,temp,temp1,temp2,temp3 =evolution_game(time_step, lattice, R, temp,temp1, temp2,temp3,ϕ, κ1, payoffs, κ, observer_coordinates,p_observer,observer_strategy_dict,a,b,σ²)
        cooprator_rates[:,i] = temp
        observer_rates[:,i] = temp1
        all_cooprator_rates[:,i] = temp2
        Pobserver_rates[:,i] = temp3
    end

    global paras = ""
    paras_list = ["p_observers","ϕ","R","κ","a","b","σ²","time_step"]
    for i in paras_list
        global paras = paras * "$i = $(eval(Symbol(i)))|"
    end


    cooprator_rates[1,end] = paras
    cooprator_rates[2:end,end] .= 0

    observer_rates[1,end] = paras
    observer_rates[2:end,end] .= 0

    all_cooprator_rates[1,end] = paras
    all_cooprator_rates[2:end,end] .= 0

    Pobserver_rates[1,end] = paras
    Pobserver_rates[2:end,end] .= 0

    writedlm("../data/p_observers_Normal_Pcooprator_rates_ave_time=$ave_time.csv",cooprator_rates)
    writedlm("../data/p_observers_Normal_Rcooprator_rates_ave_time=$ave_time.csv",observer_rates)
    writedlm("../data/p_observers_Normal_cooprator_rates_ave_time=$ave_time.csv",all_cooprator_rates)
    writedlm("../data/p_observers_Normal_rates_ave_time=$ave_time.csv",Pobserver_rates)
    
   
    