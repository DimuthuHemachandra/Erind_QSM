%Harmonize DTI data using Combat
%Works only for Baseline session for now


%CT_harmonized = get_harmonized('../Data/combat/CT/');
%PD_harmonized = get_harmonized('../Data/combat/PD/');
all_harmonized = get_harmonized('../Data/combat/volume/just_volume.csv');
function data_harmonized = get_harmonized(file_path)

   


	
        
        %M=csvread(strcat(group_path,'Baseline/',filenames{i}),1,1,[1 1 17-1 54-1]);
        M=readtable(file_path); %Reading csv files
        
        num_of_subjs = height(M)
    
        dat = M(1:num_of_subjs,2:7);
        dat =table2array(dat);
        dat = dat';
        

        batch = M(1:num_of_subjs,{'CNO'});
        batch = table2array(batch);
        
        %age = M(1:num_of_subjs,{'age_at_baseline'});
        %age = table2array(age);
        %disease = ones(1,num_of_subjs)';
        
        %mod = [age disease]
        mod=[];
  
        data_harmonized = combat(dat, batch, mod);

        data_harmonized = data_harmonized';

        M(1:num_of_subjs,2:7) = num2cell(data_harmonized);
      
        writetable(M,'../Data/combat/volume/harmonized_volume.csv','Delimiter',',')  
        %csvwrite(strcat(group_path,'combat_BL/',filenames{i}),M)


    
end
    
 
