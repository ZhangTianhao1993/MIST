function [x,fval,ppcell_mu,stdY,ageonset,thetaMu,beta,t_cn_mci,ADage,...
    e_cn,e_ad,D_ad,D_cn,Y_ad_diff,Y_cn_diff] = ...
    computeNonParaEMFixed(ptTable,hcTable,IDName,AgeName,DiagName,...
    DemoNames,featureNames,isMRI,w,knots,lambda,rho)
%--------------------------------------------------------------------------
% computeNonParaEMFixed
%
% Non-parametric Expectation–Maximization (EM) algorithm for modeling
% Alzheimer’s disease (AD) progression trajectories.
%
% This function implements a non-parametric EM-based framework to jointly
% estimate subject-specific disease onset ages and population-level
% biomarker progression trajectories along a latent disease timeline.
% The method is designed for Alzheimer’s disease but is, in principle,
% applicable to other chronic disorders characterized by monotonic
% progression. If it is to be used for other diseases, note the differences
% between the diagnostic information for that disease and AD.
%
% Core methodology:
%   1) Biomarker trajectories are modeled as monotonic cubic B-spline
%      functions of disease age (time since AD onset), with smoothness
%      enforced via an integrated squared second-derivative penalty.
%   2) Demographic effects (e.g., age, sex, education) are modeled as
%      linear covariates and regressed out from biomarker measurements.
%   3) Subject-specific AD onset ages are treated as latent variables and
%      iteratively estimated together with trajectory parameters using an
%      EM-like alternating optimization scheme.
%   4) Clinical diagnosis information (CN/MCI/AD) is incorporated through
%      a diagnosis-consistency loss that constrains estimated disease ages
%      to be compatible with observed diagnostic categories.
%
% Algorithm overview:
%   E-step (latent variable update):
%       Given current estimates of biomarker trajectories and demographic
%       effects, update each subject’s AD onset age by minimizing the
%       objective function.
%
%   M-step (parameter update):
%       Given updated AD onset ages, estimate:
%         - Biomarker trajectory parameters (theta) using constrained
%           optimization with monotonicity constraints,
%         - Demographic regression coefficients (beta),
%         - The CN-to-MCI transition point on the disease timeline.
%
% The procedure iterates until convergence of AD onset estimates.
%
% Inputs:
%   ptTable       - Table containing patient longitudinal data.
%   hcTable       - Table containing healthy subjects longitudinal data.
%   The input table should contain at least the following variables.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   IDName        - Variable name for subject identifier (Cell or String).
%   AgeName       - Variable name for age (Cell or String).
%   DiagName      - Variable name for clinical diagnosis (Cell or
%   String).(3 for Dementia, 2 for MCI, 1 for CN)
%   DemoNames     - Variable names for demographic covariates. (Cell)
%   featureNames  - Variable names of biomarkers/features. (Cell)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   isMRI         - Binary vector indicating which features are MRI-based.
%   w             - Weight vector for different biomarkers.
%   knots         - Knot locations for cubic B-spline basis.
%   lambda        - Smoothness regularization parameter.
%   rho           - Balance parameter between data fit and diagnostic info
%                   (higher rho = more weight on diagnostic information).
%
% Outputs:
%   x             - Concatenated vector of estimated parameters.
%   fval          - Final value of the objective function.
%   ppcell_mu     - Cell array of fitted biomarker trajectory splines.
%   stdY          - Feature-wise standard deviations used for normalization.
%   ageonset      - Estimated AD onset age for each subject.
%   thetaMu       - Estimated spline coefficients for biomarker trajectories.
%   beta          - Estimated demographic regression coefficients.
%   t_cn_mci      - Estimated disease-age boundary between CN and MCI.
%   ADage         - AD age (time since dementia onset) for each visit.
%   e_cn, e_ad    - Residuals for CN and AD groups, respectively.
%   D_ad, D_cn    - Demographic design matrices.
%   Y_ad_diff,
%   Y_cn_diff     - Demographically adjusted biomarker data.
%
% Notes:
%   - Monotonicity of biomarker trajectories is enforced via linear
%     inequality constraints on spline coefficients.
%
% Author:
%   Tianhao Zhang (Zhang Tianhao)
%
% Date:
%   January 05, 2026
%--------------------------------------------------------------------------
alpha = 0;
% alpha, as a parameter, determines whether to further penalize 
% AD ages during Dementia follow-up that are less than the boundary between
% CN and MCI, and follow-up ages of CN states that are greater than 0.

x = 0;
timerVal = tic;
Y_ad_raw = ptTable(:,featureNames).Variables;
Y_cn_raw = hcTable(:,featureNames).Variables;
DX_num = ptTable(:,DiagName).Variables;
r = mean(Y_ad_raw,'omitnan') - mean(Y_cn_raw,'omitnan') ;
mono = 2*(r>0)-1;
Y_ad = Y_ad_raw.*mono;
Y_cn = Y_cn_raw.*mono;

% This results in smaller data distribution bias, thus facilitating calculation.
Y = [Y_cn;Y_ad];
stdY = std(Y,'omitnan');
Y_ad = (Y_ad)./stdY;
Y_cn = (Y_cn)./stdY;


% Used to assess the impact of different demographics on characteristics.
D_ad = ptTable(:,DemoNames).Variables;
D_cn = hcTable(:,DemoNames).Variables;
D = [D_cn;D_ad];
meanD = mean(D,'omitnan');
D_ad = (D_ad)./meanD;
D_cn = (D_cn)./meanD;

D_ad = [ones(size(ptTable,1),1),D_ad];
D_cn = [ones(size(hcTable,1),1),D_cn];
D_ad(isnan(D_ad)) = 0;
D_cn(isnan(D_cn)) = 0;
ID = ptTable(:,IDName).Variables;
[UID,ia,ic] = unique(ID);
AGE = ptTable(:,AgeName).Variables;
AGE_bl = AGE(ia);
year2bl = AGE - AGE_bl(ic);

omega = BaseFunMatrix(knots);
feaNum = size(Y_ad,2);
pointNum = length(knots);

[adNum,demoNum] = size(D_ad);

weight_cn_datanum = sum(~isnan(Y_cn).*w,'all');
weight_ad_datanum = sum(~isnan(Y_ad).*w,'all');
%% ========================================================================
%  INITIALIZE DISEASE ONSET ESTIMATES
% =========================================================================
% Strategy: Use diagnostic information to estimate initial onset ages
% - Subjects with stable AD diagnosis: use midpoint of transition
% - Early AD at baseline: assume onset 5 years before baseline
% - MCI at baseline: assume onset 5 years after baseline  
% - CN at baseline: assume onset 10 years after baseline
followAgeonset = NaN(length(UID),2);
for i=1:length(UID)
    indxi = ic == i;
    Diagi = DX_num(indxi);
    Agei = AGE(indxi);
    Agei(isnan(Diagi)) = [];
    Diagi(isnan(Diagi)) = [];
    if Diagi(end) == 3
        firstADindx = find(Diagi == 3,1);
        if Diagi(1)~=3 && all(Diagi(firstADindx:end) == 3) && issorted(Diagi,'ascend')
            ageonseti = mean(Agei(firstADindx-1:firstADindx)); 
            %followAgeonset(i,1) = UID(i);
            followAgeonset(i,1) = ageonseti;
            followAgeonset(i,2) = 1;
        elseif Diagi(1) == 3
            followAgeonset(i,1) = AGE_bl(i)-5;
            followAgeonset(i,2) = 0;
        elseif Diagi(1) == 2
            followAgeonset(i,1) = AGE_bl(i)+5;
            followAgeonset(i,2) = 0;
        else
            followAgeonset(i,1) = AGE_bl(i)+10;
            followAgeonset(i,2) = 0;
        end
    elseif Diagi(1) == 3
        followAgeonset(i,1) = AGE_bl(i)-5;
        followAgeonset(i,2) = 0;
    elseif Diagi(1) == 2
        followAgeonset(i,1) = AGE_bl(i)+5;
        followAgeonset(i,2) = 0;
    else
        followAgeonset(i,1) = AGE_bl(i)+10;
        followAgeonset(i,2) = 0;
            %flag = flag + 1;
    end
end
%% ========================================================================
%  MAIN EM ALGORITHM LOOP
% =========================================================================
% Alternates between:
% 1. Fixing onset ages, optimizing progression curves and demographic effects
% 2. Fixing curves, optimizing onset ages for each subject

ageonset = followAgeonset(:,1);
while 1
 %% ====================================================================
    %  M-STEP: OPTIMIZE PARAMETERS GIVEN CURRENT ONSET ESTIMATES
    % =====================================================================
    z = find(year2bl==0);
    c = diff([z;numel(year2bl)+1]);
    ageonset_allsub = repelem(ageonset,c);
    ADage = AGE - ageonset_allsub;
    % Initialize parameter vector on first iteration
    if isequal(x,0)
        Dfilled = fillmissing(D_cn,'nearest');
        diffY = mean(Y_ad,"omitnan") - mean(Y_cn,"omitnan");
        %Y = [Y_cn;Y_ad];
        Yfilled = fillmissing(Y_cn,"nearest");
        beta = Dfilled\Yfilled;
        beta(end,isMRI==0) = 0;
        thetaMu = [zeros(3,feaNum);ones(pointNum-1,feaNum).*diffY];
        t_cn_mci = -6;
        x = [thetaMu(:);beta(:);t_cn_mci];
    end
    p1 = (pointNum+2)*feaNum;
    thetaMu = reshape(x(1:p1),[pointNum+2,feaNum]);
    p2 = p1+demoNum*feaNum;
    beta = reshape(x(p1+1:p2),[demoNum,feaNum]);
    t_cn_mci = x(end);
    options1 = optimoptions(@fmincon,'Display','none','MaxIterations',30,...
        'MaxFunctionEvaluations',10^9,...
    'UseParallel',false,'OptimalityTolerance',1e-6,'StepTolerance',1e-6,...
    'Algorithm','sqp','FiniteDifferenceType','forward');
    parfor i=1:feaNum
        betai = beta(:,i);
        y_ad_i = Y_ad(:,i);
        thetaMui = thetaMu(:,i);
        xi0 = [thetaMui(:);betai(:)];
        y_cn_i = Y_cn(:,i);
        isMRIi = isMRI(i);
        xi= fmincon(@(x)objfunFixAgeonset(x,y_ad_i,y_cn_i,D_ad,D_cn,ADage,...
            knots,omega,lambda,weight_cn_datanum,weight_ad_datanum),...
        xi0,[],[],[],[],[],[],@(x)myconFixAgeonset(x,pointNum,D_ad,isMRIi),options1);
        thetaMu(:,i) = xi(1:pointNum+2);
        beta(:,i) = xi(pointNum+3:end);
    end
     t_cn_mci = fmincon(@(x)DXloss(ADage,DX_num,x,DX_num,alpha),...
        t_cn_mci,[],[],[],[],[],[],[],options1);
    x_new = [thetaMu(:);beta(:);t_cn_mci];
    x = x_new;
    %% ====================================================================
    %  E-STEP: OPTIMIZE ONSET AGES GIVEN CURRENT PARAMETERS
    % =====================================================================
    p1 = (pointNum+2)*feaNum;
    thetaMu = reshape(x(1:p1),[pointNum+2,feaNum]);
    p2 = p1+demoNum*feaNum;
    beta = reshape(x(p1+1:p2),[demoNum,feaNum]);
    t_cn_mci = x(end);
    [~,~,ia] = unique(ID);
    fxi_mu = fnxtr(spmak(augknt(knots,4),thetaMu'),2);
    ageonset_new = ageonset;
    parfor i=1:length(unique(ID))
        if followAgeonset(i,2) == 0
            indx = ia == i;
            Y_adi = Y_ad(indx,:);
            D_adi = D_ad(indx,:);
            AGEi = AGE(indx);
            DX_numi = DX_num(indx); 
            minAD = -40;
            maxAD = 10;
            t = fminbnd(@(x)objfunFixTheta(x,Y_adi,D_adi,AGEi,beta,t_cn_mci,...
                DX_numi,rho,alpha,w,weight_ad_datanum,fxi_mu,DX_num),...
                AGEi(1)-maxAD,AGEi(1)-minAD);
            ageonset_new(i) = t;
            diffval = objfunFixTheta(t,Y_adi,D_adi,AGEi,beta,t_cn_mci,...
                DX_numi,rho,alpha,w,weight_ad_datanum,fxi_mu,DX_num) - ...
                objfunFixTheta(ageonset(i),Y_adi,D_adi,AGEi,beta,t_cn_mci,...
                DX_numi,rho,alpha,w,weight_ad_datanum,fxi_mu,DX_num);
            if diffval>0
               ageonset_new(i) = ageonset(i);
            end
        end
    end
    x_full = [ageonset_new(:);thetaMu(:);beta(:);t_cn_mci];
    %% ====================================================================
    %  CONVERGENCE CHECK
    % =====================================================================
    if max(abs(ageonset_new - ageonset))>0.1
        ageonset = ageonset_new;
    else
        x_full = [ageonset_new;x];
        fval = objectfunction(x_full,Y_ad,Y_cn,D_ad,D_cn,AGE,knots,omega,...
            year2bl,DX_num,lambda,rho,alpha,w,weight_cn_datanum,weight_ad_datanum);
        fprintf('final fval = %f time = %f\n',fval,toc(timerVal));
        break
    end
end
%% ========================================================================
%  FINAL PARAMETER EXTRACTION AND DENORMALIZATION
% =========================================================================
x = x_full;

subNum = sum(year2bl==0); 

p1 = subNum;
ageonset = x(1:p1);
p2 = p1+(pointNum+2)*feaNum;
thetaMu = reshape(x(p1+1:p2),[pointNum+2,feaNum]).*mono.*stdY;
p3 = p2+demoNum*feaNum;
beta = reshape(x(p2+1:p3),[demoNum,feaNum]).*mono.*stdY;
t_cn_mci = x(end);
    
z = find(year2bl==0);
c = diff([z;numel(year2bl)+1]);
ageonset_allsub = repelem(ageonset,c);
ADage = (AGE - ageonset_allsub);
%% ========================================================================
%  CREATE OUTPUT STRUCTURES
% =========================================================================
ppcell_mu = cell(feaNum,1);
for i=1:feaNum
    sp_mu = spmak(augknt(knots,4),thetaMu(:,i)');
    pp_mu = fnxtr(sp_mu,2);
    ppcell_mu(i) = {pp_mu};
end
e_cn = Y_cn_raw - D_cn*beta;
Y_cn_diff = e_cn;
fxi_mu = fnxtr(spmak(augknt(knots,4),thetaMu'),2);
fxmu = fnval(fxi_mu,ADage)';
Y_ad_diff = Y_ad_raw - D_ad*beta;
e_ad = Y_ad_diff  - fxmu;
beta(2:end,:) = beta(2:end,:)./meanD';
end
%% ========================================================================
%  OBJECTIVE FUNCTION: FULL MODEL LOSS
% =========================================================================
function loss = objectfunction(x,Y_ad,Y_cn,D_ad,D_cn,AGE,knots,omega,year2bl,...
    DX_num,lambda,rho,alpha,w,weight_cn_datanum,weight_ad_datanum)
    subNum = sum(year2bl==0);
    feaNum = size(Y_ad,2); 
    pointNum = length(knots); 
    [adNum,demoNum] = size(D_ad); 
    DXtimes = sum(~isnan(DX_num));
    p1 = subNum;
    ageonset = x(1:p1);
    p2 = p1+(pointNum+2)*feaNum;
    thetaMu = reshape(x(p1+1:p2),[pointNum+2,feaNum]);
    p3 = p2+demoNum*feaNum;
    beta = reshape(x(p2+1:p3),[demoNum,feaNum]);
    t_cn_mci = x(end);
    z = find(year2bl==0);
    c = diff([z;numel(year2bl)+1]);
    ageonset_allsub = repelem(ageonset,c);
    ADage = AGE - ageonset_allsub;
    t1 = (Y_cn - D_cn*beta).^2;
    smoothing_mu = NaN(feaNum,1);
    fxi_mu = fnxtr(spmak(augknt(knots,4),thetaMu'),2);
    fxmu = fnval(fxi_mu,ADage)';

    for i=1:feaNum
        thetaMui = thetaMu(:,i);

        tempvar1 = thetaMui*thetaMui';
        smoothing_mu(i) = sum(tempvar1.*omega,"all");

    end

    t2 = (Y_ad - D_ad*beta - fxmu).^2;
    t2 = [t2;lambda*smoothing_mu'];
    loss_cn = sum(t1.*w,'all','omitnan')/weight_cn_datanum;
    loss_pt = sum(t2.*w,'all','omitnan')/weight_ad_datanum;
    loss_DX = rho*DXloss(ADage,DX_num,t_cn_mci,DX_num,alpha);
    loss = loss_cn+loss_pt+loss_DX;
end


%% ========================================================================
%  DIAGNOSTIC LOSS FUNCTION
% =========================================================================
function loss = DXloss(ADage,DX_num,t_cn_mci,DX_num_all,alpha)
      power = 3;
      w1 = (ADage<0)&(DX_num == 3);
      loss1 = sum((0-ADage(w1)).^power,'all');
      %loss1 = 0;
      w2 = (ADage>0)&(DX_num == 2);
      loss2 = sum((ADage(w2)).^power,'all');
      %loss2 = 0;
      w3 = (ADage>t_cn_mci)&(DX_num == 1);
      loss3 = sum((ADage(w3)-t_cn_mci).^power,'all');
      %loss3 = 0;
      w4 = (ADage<t_cn_mci)& (DX_num == 2);
      loss4 = sum((t_cn_mci-ADage(w4)).^power,'all');
      %loss4 = 0;
      w5 = (ADage<t_cn_mci)&(DX_num == 3);
      loss5 = alpha*sum((t_cn_mci-ADage(w5)).^power,'all');
      w6 = (ADage>0)&(DX_num == 1);
      loss6 = alpha*sum((ADage(w6)).^power,'all');
      loss = (loss1+loss5)/sum(DX_num_all==3)+(loss3+loss6)/sum(DX_num_all==1)+...
          (loss2+loss4)/sum(DX_num_all==2);
end
%% ========================================================================
%  SPLINE BASIS FUNCTION MATRIX COMPUTATION
% =========================================================================
function omega = BaseFunMatrix(knots)
    thetaNum = length(knots)+2;
    knots = augknt(knots,4);
    NFun = cell(thetaNum,1);
    d2NFun = cell(thetaNum,1);
    %N = NaN(subNum,thetaNum);
    omega = NaN(thetaNum,thetaNum);
    for i=1:thetaNum
        NFun{i} = fnxtr(bspline(knots(i:i+4)),0);
        d2NFun{i} = fnder(NFun{i},2);
    end
    for i=1:thetaNum
        for j=1:thetaNum
            if abs(i-j)<4
                ddfx = fncmb(d2NFun{i},'*',d2NFun{j});
                omega(i,j) = diff(fnval(fnint(ddfx),[ddfx.breaks(1),ddfx.breaks(end)]));
            else
                omega(i,j) = 0;
            end
        end
    end
end
%% ========================================================================
%  OBJECTIVE FUNCTION FOR FIXED ONSET (M-STEP)
% =========================================================================
function loss = objfunFixAgeonset(x,y_ad_i,y_cn_i,D_ad,D_cn,ADage,knots,omega,...
    lambda,weight_cn_datanum,weight_ad_datanum)
    pointNum = length(knots); 
    [adNum,demoNum] = size(D_ad); 

    p1 = pointNum+2;
    thetaMu = x(1:p1);
    p2 = p1+demoNum;
    beta = x(p1+1:p2);

    t1 = (y_cn_i - D_cn*beta).^2;
    fxi_mu = fnxtr(spmak(augknt(knots,4),thetaMu'),2);
    fxmu = fnval(fxi_mu,ADage)';
    tempvar1 = thetaMu*thetaMu';
    smoothing_mu = sum(tempvar1.*omega,"all");
    t2 = (y_ad_i - D_ad*beta - fxmu').^2;
    t2 = [t2;lambda*smoothing_mu'];
    loss_cn = weight_ad_datanum*sum(t1,'omitnan');
    loss_pt = weight_cn_datanum*sum(t2,'omitnan');
    loss = loss_cn+loss_pt;
end
%% ========================================================================
%  CONSTRAINTS FOR M-STEP OPTIMIZATION
% =========================================================================
function [c,ceq] = myconFixAgeonset(x,pointNum,D_ad,isMRIi)
    [adNum,demoNum] = size(D_ad);
    %rNum = size(R_ad,2);

    p1 = pointNum+2;
    thetaMu = x(1:p1);
    p2 = p1+demoNum;
    beta = x(p1+1:p2);
    A = diag(ones(pointNum+2,1)) + diag(-ones(pointNum+1,1),1);
    A(pointNum+2,:) = [];
    c = A*thetaMu;
    if isMRIi == 0
        ceq = [thetaMu(1);thetaMu(2);thetaMu(3);beta(end)];
    else
        ceq = [thetaMu(1);thetaMu(2);thetaMu(3)];
    end
end
%% ========================================================================
%  OBJECTIVE FUNCTION FOR FIXED PARAMETERS (E-STEP)
% =========================================================================
function loss = objfunFixTheta(x,Y_adi,D_adi,AGEi,beta,t_cn_mci,DX_numi,...
    rho,alpha,w,weight_ad_datanum,fxi_mu,DX_num)
    ADage = AGEi - x;
    fxmu = fnval(fxi_mu,ADage)';
    t2 = (Y_adi - D_adi*beta - fxmu).^2;
    loss_pt = sum(t2.*w,'all','omitnan')/weight_ad_datanum;
    loss_DX = rho*DXloss(ADage,DX_numi,t_cn_mci,DX_num,alpha);
    loss = loss_pt+loss_DX;
end
