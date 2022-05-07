%Harmonize DTI data using Combat
%Works only for Baseline session for now



VTASN_harmonized = get_harmonized('../Data/combat/all/');

function data_harmonized = get_harmonized(group_path)

    dinfo = dir(strcat(group_path,'PPMI_with_VTASN/PD/*.csv')); %Reading paths to csv files generated by combat.py
    filenames = {dinfo.name};
    n = length(filenames)

	for i=1:n
        
        %M=csvread(strcat(group_path,'Baseline/',filenames{i}),1,1,[1 1 17-1 54-1]);
        M=readtable(strcat(group_path,'PPMI_with_VTASN/PD/',filenames{i})); %Reading csv files
        
        num_of_subjs = height(M)
        filenames{i}
    
        dat = M(1:num_of_subjs,2:15);
        dat =table2array(dat);
        dat = dat';
        

        batch = M(1:num_of_subjs,{'CNO'});
        batch = table2array(batch);
        
        age = M(1:num_of_subjs,{'age_at_baseline'});
        age = table2array(age);
        %disease = ones(1,num_of_subjs)';
        %disease = M(1:num_of_subjs,{'group_ID'});
        %disease = table2array(disease);
        disease = ones(1,num_of_subjs)';
        mod = [age disease]
  
        data_harmonized = combat(dat, batch, mod);

        data_harmonized = data_harmonized';

        M(1:num_of_subjs,2:15) = num2cell(data_harmonized);
      
        writetable(M,strcat(group_path,'combat_PPMI_with_VTASN/PD/',filenames{i}),'Delimiter',',')  
        %csvwrite(strcat(group_path,'combat_BL/',filenames{i}),M)


    end
end
    
 
