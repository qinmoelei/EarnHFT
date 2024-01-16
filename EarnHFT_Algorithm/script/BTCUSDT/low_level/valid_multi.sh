function getnextdir_1(){
    counter=0
    for element in `ls $1 | sort -t '_' -k2 -n`
    do  
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ] ;then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ] ;then
            cuda_number=2
        else 
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/pick/BTCUSDT/beta_-10/position_0_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/low_level/test_ddqn.py \
        --test_path $target_path --initial_action 0 --test_df_path data/BTCUSDT/valid  \
        >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}






function getnextdir_2(){
    counter=0
    for element in `ls $1 | sort -t '_' -k2 -n`
    do  
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ] ;then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ] ;then
            cuda_number=2
        else 
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/pick/BTCUSDT/beta_-10/position_1_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/low_level/test_ddqn.py \
        --test_path $target_path --initial_action 1 --test_df_path data/BTCUSDT/valid  \
        >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}



function getnextdir_3(){
    counter=0
    for element in `ls $1 | sort -t '_' -k2 -n`
    do  
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ] ;then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ] ;then
            cuda_number=2
        else 
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/pick/BTCUSDT/beta_-10/position_2_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/low_level/test_ddqn.py \
        --test_path $target_path --initial_action 2 --test_df_path data/BTCUSDT/valid  \
        >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}


function getnextdir_4(){
    counter=0
    for element in `ls $1 | sort -t '_' -k2 -n`
    do  
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ] ;then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ] ;then
            cuda_number=2
        else 
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/pick/BTCUSDT/beta_-10/position_3_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/low_level/test_ddqn.py \
        --test_path $target_path --initial_action 3 --test_df_path data/BTCUSDT/valid  \
        >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}


function getnextdir_5(){
    counter=0
    for element in `ls $1 | sort -t '_' -k2 -n`
    do  
        if [ "$(basename $element)" = "log" ]; then
            continue
        fi
        epoch=$(basename $element | cut -d'_' -f2)
        if [ $((counter % 4)) -eq 0 ]; then
            cuda_number=0
        elif [ $((counter % 4)) -eq 1 ] ;then
            cuda_number=1
        elif [ $((counter % 4)) -eq 2 ] ;then
            cuda_number=2
        else 
            cuda_number=3
        fi
        echo $counter
        echo $epoch
        echo $element

        target_path=$1"/"$element
        log_filename="log/pick/BTCUSDT/beta_-10/position_4_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/low_level/test_ddqn.py \
        --test_path $target_path --initial_action 4 --test_df_path data/BTCUSDT/valid  \
        >$log_filename 2>&1 &
        last_pid=$!
        counter=$((counter + 1))
        if [ $counter -eq 50 ]; then
            break
        fi
    done
    wait $last_pid
}








root_dir_1="result_risk/BTCUSDT/beta_-10.0_risk_bond_0.1/seed_12345"
root_dir_2="result_risk/BTCUSDT/beta_-90.0_risk_bond_0.1/seed_12345"
root_dir_3="result_risk/BTCUSDT/beta_30.0_risk_bond_0.1/seed_12345"
root_dir_4="result_risk/BTCUSDT/beta_100.0_risk_bond_0.1/seed_12345"


getnextdir_1 $root_dir_1
getnextdir_2 $root_dir_1
getnextdir_3 $root_dir_1
getnextdir_4 $root_dir_1
getnextdir_5 $root_dir_1


getnextdir_1 $root_dir_2
getnextdir_2 $root_dir_2
getnextdir_3 $root_dir_2
getnextdir_4 $root_dir_2
getnextdir_5 $root_dir_2



getnextdir_1 $root_dir_3
getnextdir_2 $root_dir_3
getnextdir_3 $root_dir_3
getnextdir_4 $root_dir_3
getnextdir_5 $root_dir_3


getnextdir_1 $root_dir_4
getnextdir_2 $root_dir_4
getnextdir_3 $root_dir_4
getnextdir_4 $root_dir_4
getnextdir_5 $root_dir_4

