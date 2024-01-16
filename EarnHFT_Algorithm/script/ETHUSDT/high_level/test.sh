function getnextdir(){
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
        log_filename="log/test/ETHUSDT/test_$element.log"
        CUDA_VISIBLE_DEVICES=$cuda_number nohup python RL/agent/high_level/test_dqn_position.py \
        --test_path $target_path --dataset_name ETHUSDT --valid_data_path data/ETHUSDT/valid.feather --test_data_path data/ETHUSDT/test.feather --max_holding_number 0.1 \
        >$log_filename 2>&1 &
        counter=$((counter + 1))
        if [ $counter -eq 80 ]; then
            break
        fi
    done
}


root_dir_1="result_risk/ETHUSDT/high_level/seed_12345"


getnextdir $root_dir_1