function preTable = UsingMultiFeaturesPredictADage(newflupAgeTable,colnames,DemoNames,DiagName,...
    knots_model,XthetaMu_model,Xbeta_model,t_cn_mci_model,rho_model,Xw_model,XstdY_model)
%--------------------------------------------------------------------------
% UsingMultiFeaturesPredictADage
%
% Individual-level prediction of Alzheimer’s disease age using multiple
% biomarkers and a pretrained progression model.
%
% This function applies a previously trained non-parametric disease
% progression model to new follow-up data in order to estimate each
% subject’s disease age (time relative to dementia onset). All model parameters
% (trajectory shapes, demographic effects, transition points, and
% regularization weights) are assumed to be fixed and obtained from an
% external training cohort (e.g., ADNI).
%
% Inputs:
%   newflupAgeTable   - Table containing new subject data.
%   colnames – Biomarker feature identifiers, specified either as variable
%           names or as numeric column indices of the input table.
%   DemoNames         - Variable names for demographic covariates.
%   DiagName          - Variable name for clinical diagnosis.
%   knots_model       - Knot locations from the pretrained spline model.
%   XthetaMu_model    - Pretrained spline coefficients (biomarker trajectories).
%   Xbeta_model       - Pretrained demographic regression coefficients.
%   t_cn_mci_model    - Pretrained CN-to-MCI transition point.
%   rho_model         - Pretrained diagnosis-consistency weight.
%   Xw_model          - Feature-specific weights.
%   XstdY_model       - Pretrained Feature-wise standard deviations for normalization.
% Outputs:
%   preTable          - Input table augmented with predicted AD disease age.
%
% Notes:
%   - Prediction is performed independently for each subject/visit.
%   - Subjects whose predicted disease age contradicts their diagnosis
%     category are projected back to the diagnostic boundary (ADage = 0).
%
% Author:
%   Tianhao Zhang (Zhang Tianhao)
%
% Date:
%   January 05, 2026
%--------------------------------------------------------------------------

preTable = newflupAgeTable;
n = size(preTable,1);
predictADage = NaN(n,1);
    for i=1:n
        Y_adi = newflupAgeTable(i,colnames).Variables;
        diagi = newflupAgeTable(i,DiagName).Variables;
        centiloidi = newflupAgeTable(i,'CENTILOIDS').Variables;
        %Y_adi([1:4,9:12]) = NaN;
        if sum(~isnan(Y_adi)) == length(Y_adi)
            if ~(diagi ==1 && centiloidi < 20.1)
            D_adi = [1,newflupAgeTable(i,DemoNames).Variables];
            D_adi(isnan(D_adi)) = 0;
            DX_numi = newflupAgeTable(i,DiagName).Variables;
            predictADage(i) = predictTimeFromADonset(knots_model,XthetaMu_model,...
                Xbeta_model,t_cn_mci_model,rho_model,Y_adi,D_adi,DX_numi,Xw_model,XstdY_model);
            if ((diagi == 1 || diagi == 2) && predictADage(i)>0) || (diagi == 3 && predictADage(i)<0)
                predictADage(i) = 0;
            end
            end
        end
    end
preTable = addvars(newflupAgeTable,predictADage);
end
function predictTime = predictTimeFromADonset(knots_model,thetaMu_model,beta_model,t_cn_mci_model,rho_model,Y_adi,D_adi,DX_numi,w_model,stdY_model)
%--------------------------------------------------------------------------
% predictTimeFromADonset
%
% Estimate disease age for a single subject given fixed model parameters.
%
% This function estimates the disease age (time from AD onset) that best
% aligns a subject’s biomarker profile with pretrained progression
% trajectories. The estimate is obtained by minimizing a diagnosis-aware
% objective function over a bounded disease-age interval.
%
% The optimization is one-dimensional and performed using Brent’s method
% (fminbnd).
%--------------------------------------------------------------------------
fxi_mu_model = fnxtr(spmak(augknt(knots_model,4),thetaMu_model'),2);
alpha = 0;
predictTime = fminbnd(@(x)objfunFixTheta(x,Y_adi,D_adi,beta_model,...
    t_cn_mci_model,DX_numi,rho_model,alpha,w_model,fxi_mu_model,stdY_model),-15,10);
end

function loss = objfunFixTheta(ADage,Y_adi,D_adi,beta,t_cn_mci,DX_numi,rho,alpha,w,fxi_mu,stdY)
    loss_DX = rho*DXloss(ADage,DX_numi,t_cn_mci,alpha);
    fxmu = fnval(fxi_mu,ADage)';
    t2 = ((Y_adi - D_adi*beta - fxmu)./stdY).^2;
    if length(Y_adi)>=84
        Y_adi_predict = D_adi*beta + fxmu;
        R = corr(Y_adi(1:84)',Y_adi_predict(1:84)');
        imgLoss = sum(t2(1:84).*w(1:84),'all','omitnan');
        if length(Y_adi)>84
            cogLoss = sum(t2(85:end).*w(85:end),'all','omitnan');
            loss_pt = (imgLoss+cogLoss);%*sum(t2(85:end).*w(85:end),'all','omitnan');
            loss = (((1-R)*loss_pt+loss_DX));
        else
            loss_pt = imgLoss;
            loss = (((1-R)*loss_pt+loss_DX));
        end
    else
        loss_pt = sum(t2.*w,'all','omitnan');
        loss = loss_pt+loss_DX;
    end
end
function loss = DXloss(ADage,DX_num,t_cn_mci,alpha)
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
      loss = loss1+loss2+loss3+loss4+loss5+loss6;
end