function [lambda,rho,lossCub,hyperparaMatrix] = getHyperpara(ptTable,hcTable,...
    IDName,DiagName,AgeName,DemoNames,featureNames,isMRI,w,knots,L,lambdalist,rholist,seednum)
%--------------------------------------------------------------------------
% getHyperpara
%
% Cross-validated hyperparameter selection for the non-parametric EM
% disease progression model.
%
% This function performs grid-search–based cross-validation to determine
% optimal values of the smoothness regularization parameter (lambda) and
% the diagnosis-consistency weight (rho) used in the non-parametric EM
% framework for Alzheimer’s disease progression modeling.
%
% Methodological background:
%   In the proposed model, lambda controls the smoothness of biomarker
%   progression trajectories by penalizing the integrated squared second
%   derivative of spline functions, whereas rho controls the strength of
%   the diagnosis-consistency loss that enforces agreement between
%   estimated disease age and observed clinical diagnoses (CN/MCI/AD).
%
%   To balance trajectory smoothness, predictive accuracy, and diagnostic
%   consistency, both hyperparameters are selected via subject-level
%   cross-validation on individuals with AD-like progression patterns.
%
% Cross-validation procedure:
%   1) Subjects with a monotonic diagnostic transition to AD are identified.
%   2) These subjects are split into L folds at the subject level.
%   3) For each (lambda, rho) pair and each fold:
%        a) The model is trained on L–1 folds using computeNonParaEMFixed.
%        b) Biomarker trajectories and demographic effects are estimated.
%        c) Prediction loss is evaluated on the held-out fold by comparing
%           predicted biomarker values against observations at known or
%           estimated disease ages.
%   4) Losses are aggregated across folds.
%   5) The (lambda, rho) pair minimizing the total cross-validated loss
%      is selected.
%
% Inputs:
%   ptTable   - Table containing longitudinal data of patient subjects
%   hcTable    - Table containing hc reference data.
%   IDName        - Variable name for subject identifier.
%   DiagName      - Variable name for clinical diagnosis (numeric-coded).
%   AgeName       - Variable name for age at visit.
%   DemoNames     - Variable names for demographic covariates.
%   featureNames  - Variable names of biomarkers/features.
%   isMRI         - Binary vector indicating which features are MRI-based.
%   w             - Feature-specific weights.
%   knots         - Knot locations for cubic B-spline basis.
%   L             - Number of cross-validation folds.
%   lambdalist    - Candidate values for smoothness parameter lambda.
%   rholist       - Candidate values for diagnosis-consistency weight rho.
%   seednum       - Random seed for reproducible data splitting.
%
% Outputs:
%   lambda            - Selected optimal smoothness parameter.
%   rho               - Selected optimal diagnosis-consistency weight.
%   lossCub           - 3D array of cross-validation losses
%                       (lambda × rho × fold).
%   hyperparaMatrix   - Cell array storing all tested (lambda, rho) pairs.
%
% Notes:
%   - Cross-validation is performed at the subject level to avoid data
%     leakage across visits of the same individual.
%   - CN data are always included in model training to anchor biomarker
%     baselines.
%   - Feature-wise normalization is performed using statistics computed
%     from the full dataset.
%
% Author:
%   Tianhao Zhang (Zhang Tianhao)
%
% Date:
%   January 05, 2026
%--------------------------------------------------------------------------
    [trainTable,validTable,validID,validAge] = splitTable(ptTable,IDName,DiagName,AgeName,L,seednum);
    
    Y_ad = ptTable(:,featureNames).Variables;
    Y_cn = hcTable(:,featureNames).Variables;
    Y = [Y_cn;Y_ad];
    stdY = std(Y,'omitnan');
    n = length(lambdalist);
    m = length(rholist);
    lossCub = NaN(n,m,L);
    hyperparaMatrix = cell(n,m);
    for i=1:n
        for j=1:m
           hyperparaMatrix(i,j) = {[lambdalist(i),rholist(j)]};
        end
    end
    for k=1:L
        lossMatrix = NaN(n,m);
        for i=1:n
            for j=1:m
                lambda = lambdalist(i);
                rho = rholist(j);
                [~,~,~,~,~,thetaMu,beta,t_cn_mci] = ...
                    computeNonParaEMFixed(trainTable{k},hcTable,IDName,AgeName,DiagName,...
                DemoNames,featureNames,isMRI,w,knots,lambda,rho);
                [loss,~,~] = predFeaRealADage(validTable{k},validID{k},validAge{k},beta,DemoNames,...
                stdY,knots,thetaMu,featureNames,w);
                lossMatrix(i,j) = loss;
                fprintf('lambda = %f, rho = %f, loss = %f, t_cn_mci = %f\n',lambda,rho,loss,t_cn_mci);
            end
        end
        lossCub(:,:,k) = lossMatrix;
        fprintf('k = %d\n',k);
    end
    [~,I] = min(sum(lossCub,3),[],'all');
    t= cell2mat(hyperparaMatrix(I));
    lambda = t(1);
    rho = t(2);
end

function [trainTable,validTable,validID,validAge] = ...
    splitTable(ADTable,IDName,DiagName,AgeName,foldnum,seednum)
%--------------------------------------------------------------------------
% splitTable
%
% Subject-level cross-validation split for AD-like longitudinal data.
%
% This function partitions an AD-like longitudinal dataset into training
% and validation folds for cross-validation. Only subjects exhibiting a
% monotonic diagnostic progression to Alzheimer’s disease (AD) are used
% for validation splitting, ensuring well-defined disease onset ordering.
%
% Key steps:
%   1) Identify subjects whose diagnosis transitions monotonically to AD
%      (e.g., CN/MCI → AD without reversion).
%   2) Exclude subjects with non-monotonic or inconsistent diagnostic
%      trajectories.
%   3) Estimate an approximate AD onset age for each eligible subject
%      based on the transition time to AD.
%   4) Randomly split subjects into the specified number of folds.
%   5) Construct subject-level training and validation tables.
%
% Inputs:
%   ADTable    - Table containing longitudinal AD-like subject data.
%   IDName     - Variable name for subject identifier.
%   DiagName   - Variable name for clinical diagnosis (numeric-coded).
%   AgeName    - Variable name for age at visit.
%   foldnum    - Number of cross-validation folds.
%   seednum    - Random seed for reproducible splitting.
%
% Outputs:
%   trainTable - Cell array of training tables for each fold.
%   validTable - Cell array of validation tables for each fold.
%   validID    - Cell array of subject IDs in each validation fold.
%   validAge   - Cell array of estimated AD onset ages for validation subjects.
%
% Notes:
%   - Splitting is performed at the subject level rather than the visit
%     level to preserve longitudinal integrity.
%   - Only subjects with reliable AD conversion patterns are used for
%     validation to ensure meaningful evaluation of disease-age prediction.
%
% Author:
%   Tianhao Zhang (Zhang Tianhao)
%
% Date:
%   January 05, 2026
%--------------------------------------------------------------------------
ID = ADTable(:,IDName).Variables;
Diag = ADTable(:,DiagName).Variables;
Age = ADTable(:,AgeName).Variables;
flag = 1;
[UID,ia,ic] = unique(ID);
for i=1:length(ia)
    indxi = ic == i;
    Diagi = Diag(indxi);
    Agei = Age(indxi);
    Agei(isnan(Diagi)) = [];
    Diagi(isnan(Diagi)) = [];
    if Diagi(end) == 3 
        firstADindx = find(Diagi == 3,1);
        if Diagi(1)~=3 && all(Diagi(firstADindx:end) == 3) && issorted(Diagi,'ascend')
            ageonseti = mean(Agei(firstADindx-1:firstADindx)); 
            followAgeonset(flag,1) = UID(i);
            followAgeonset(flag,2) = ageonseti;
            flag = flag + 1;
        end
    end
end
validID = cell(foldnum,1);
validAge = cell(foldnum,1);
validTable = cell(foldnum,1);
trainTable = cell(foldnum,1);
n = length(UID);
Uageonset = NaN(n,1);
for i=1:n
     Indx = followAgeonset(:,1) == UID(i);
    if ~all(Indx==0)
        Uageonset(i) = followAgeonset(Indx,2);
    end
end
rand('seed',seednum);
p = randperm(n);
y = linspace(0,n,foldnum+1);
    for i=1:foldnum
        pi = p(floor(y(i)+1):floor(y(i+1)));
        validID(i) = {UID(pi,1)};
        validAge(i) = {Uageonset(pi,1)};
        validTable(i) = {ADTable(ismember(ID,validID{i}),:)};
        trainTable(i) = {ADTable(~ismember(ID,validID{i}),:)};
    end
end

function [loss,realADage,features]= predFeaRealADage(validTable,validID,validAge,beta,DemoNames,...
    stdY,knots,thetaMu,featureNames,w)
    T = table(validID,validAge);
    validTable = join(validTable,T,'LeftKeys',1,'RightKeys',1);
    realADage = validTable.AGE - validTable.validAge;
    validTable.realADage = realADage;
    features = validTable(:,featureNames).Variables;
    Demos = validTable(:,DemoNames).Variables;
    Demos = [ones(length(Demos),1),Demos];
    Demos(isnan(Demos)) = 0;
    y_ad_diff = features - Demos*beta;
    predictY = fnval(fnxtr(spmak(augknt(knots,4),thetaMu'),2),realADage);
    weight_ad_datanum = sum(~isnan(y_ad_diff).*w,'all');
    loss = sum(((y_ad_diff - predictY')./stdY).^2.*w,'all','omitnan')/weight_ad_datanum;
end