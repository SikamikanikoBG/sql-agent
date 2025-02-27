CREATE VIEW [dbo].[vBG_Direct_Production] AS 

SELECT 

		t.ID,
		t.CustomerID,
		t.CreditNumber,
		t.SentOn,
		t.ReclaimedOn,
		t.StartedOn,
		case when cast(SentOn as date)< cast(ReclaimedOn as date) THEN cast(ReclaimedOn as date) ELSE cast(SentOn as date) END as SentDateRecalc,
		t.IsReclaimed,
		t.IsApproved,
		t.Product,
		t.Region,
		t.Office,
		t.SentRegion,
		t.SentOffice,
		t.ReclaimedRegion,
		t.ReclaimedOffice,
		t.StartedByName,
		t.SentByName,
		t.Maturity,
		t.Amount AmountOrigin,
		t.AmountNewProject,
		SUM(t.Amount+ISNULL(pol.Amount,0)) Amount,
		t.AmountWithInsurance,
		t.ClauseAPrincipal,
		t.ClauseBPrincipal,
		t.CpiPrincipal,
		t.IsDirectCardOpening,
		t.MarketingAction,
		t.CreditState,
		t.Channel,
		t.ClientProfile,
		t.ClientType,
		t.MaxInstallment,
		t.MaxLimit_SIMU,
		t.FinalLimit,
		t.InitialPurposeAmount,
		t.PaymentScheme,
		t.ConsoLastYear,
		t.CounterOfferByName,
		t.CustomerType,
		t.DataSharingConsent,
		t.Note_text_Date,
		t.Note_text,
		t.Note_text_9300,
		t.Note_text_9300_Date,
		t.IsConsolidation,
		t.ConsoEligible,
		t.ConsoEligible_Full,
		t.ConsoEligible_Partial,
		t.ConsoAmountInternal,
		t.ConsolidatedAmountReal,
		t.ConsoPotentialOther,
		t.LastNotSubmitReason,
		t.MarketingConsent,
		t.ReasonUndisbursed,
		t.HasEmail,
		t.SocialMediaConsents,
		t.NIR,
		t.IRR,
		t.CustomerAge,
		t.GeneratedOutstanding,
		t.IsTelesalesApplication,
		t.BiCo,
		t.IsEsignEligibleFlag,
		t.IsEsignFlag,
		CASE WHEN pol.fkCredit is not null THEN 1 ELSE 0 end HasHomeIns,
		[HomeInsType],
		ISNULL(pol.Amount,0)  [HomeInsInsurancePremium],
		CASE WHEN t.Maturity<=24 THEN 1 ELSE 0 END EligibleHomeIns,
		CASE WHEN t.CpiPrincipal>0 THEN 1 ELSE 0 END CPIInsured,
		CASE WHEN t.AmountWithInsurance<=3000 THEN 0 ELSE 1 END 'AmountWithInsurance>3000'
		

FROM [DWH_Stage].[dbo].[BG_DIRECT_PRODUCTION] t
LEFT JOIN(select 
          fkCredit,ISNULL(pr.[Description],'') [HomeInsType] ,StartDate,Amount,
          ROW_NUMBER() over(Partition by fkCredit order by StartDate desc) row_id
		  from      [Jetix].[dbo].tHomeInsurancePolicy pol with(NOLOCK)  
		  left join [Jetix].[dbo].tHomeInsuranceProduct pr with(NOLOCK)  on pr.id = pol.fkHomeInsuranceProduct
		   ) pol on pol.fkCredit = t.ID and row_id = 1

WHERE CAST(StartedOn as date) >= DATEADD(yy,DATEDIFF(yy,0,GETDATE())-3,0)
and Channel !='Револвираща сметка' ---Petya A. 20240731: изключен е канал поради промяна в процес FastBico
group by    t.StartedOn,
			t.ID,
			t.CustomerID,
			t.CreditNumber,
			t.SentOn,
			t.ReclaimedOn,
			t.IsReclaimed,
			t.IsApproved,
			t.Product,
			t.Region,
			t.Office,
			t.SentRegion,
			t.SentOffice,
			t.ReclaimedRegion,
			t.ReclaimedOffice,
			t.StartedByName,
			t.SentByName,
			t.Maturity,
			t.Amount,
			t.AmountWithInsurance,
			t.ClauseAPrincipal,
			t.ClauseBPrincipal,
			t.CpiPrincipal,
			t.IsDirectCardOpening,
			t.MarketingAction,
			t.CreditState,
			t.Channel,
			t.ClientProfile,
			t.ClientType,
			t.MaxInstallment,
			t.MaxLimit_SIMU,
			t.FinalLimit,
			t.InitialPurposeAmount,
			t.PaymentScheme,
			t.ConsoLastYear,
			t.CounterOfferByName,
			t.CustomerType,
			t.DataSharingConsent,
			t.Note_text_Date,
			t.Note_text,
			t.Note_text_9300,
			t.Note_text_9300_Date,
			t.IsConsolidation,
			t.ConsoEligible,
			t.ConsoEligible_Full,
			t.ConsoEligible_Partial,
			t.ConsoAmountInternal,
			t.ConsolidatedAmountReal,
			t.ConsoPotentialOther,
			t.LastNotSubmitReason,
			t.MarketingConsent,
			t.ReasonUndisbursed,
			t.HasEmail,
			t.SocialMediaConsents,
			t.NIR,
			t.IRR,
			t.CustomerAge,
			t.GeneratedOutstanding,
			t.IsTelesalesApplication,
			t.BiCo,
			t.IsEsignEligibleFlag,
		    t.IsEsignFlag,
			pol.fkCredit,
		    HomeInsType,
			pol.Amount,
			t.AmountNewProject