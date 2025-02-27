----- 

CREATE VIEW [dbo].[vBG_Retail_Production] AS 

SELECT	prod.ID,
        prod.CreditNumber,
		prod.CreditState,		
        prod.StartedOn,
		prod.StartedByName,
		prod.SentOn,
		prod.CustomerID,
		prod.DatePayment,
		ISNULL(ISNULL(uidname.Username,uname.Username),'') SentByName,
		prod.EmployeeUID,
		prod.EmployeeType,
		prod.ReclaimedOn,
		prod.IsReclaimed,
		prod.IsApproved,
     	prod.ReclaimedByName,
		prod.ReclaimedEmployeeUID, 
		prod.BisinessDevelopmentManager,
		prod.SalesRegion,
		prod.MerchantPartnerGroup,
		prod.MerchantPartnerName,
		prod.MerchantMarket,
		prod.MPCode,
		prod.StoreName,
		prod.ShopType,
		prod.CreditType,
		prod.Channel,
		prod.IRR,
		prod.Principal,
		prod.Amount,
		prod.[Maturity*Principal],
		prod.GeneratedOutstanding,
		prod.[Interest&Fees],
		prod.ClientType,
		prod.HasInsurance,
		prod.LoansWithCPI,
		prod.LoansWithCPIEligibleForFullPack,
		prod.LoansWithCPINotEligibleForFullPack,
		prod.LoansWithFullPack,
		prod.HasEmail,
		prod.MarketingConsent,
		prod.DataSharingConsent,
		prod.SocialMediaConsents,
		prod.IsEligibleForCreditCard,
		prod.IsEsignFlag,
		prod.IsEsignEligibleFlag,
		prod.ACTIVATED_M2_CC,
		prod.IsMobileVisit,
		IsVRC.IsVoiceRecording,
		prod.PaymentScheme,
		prod.PaymentSchemeName
		,ISNULL(mobile.IsMobileVisitSent,0) IsMobileVisitSent
		,ISNULL(el.DTS_Eligible,0) DTS_Eligible
		,CASE WHEN PDDsigned.PDDSignedToQuickSignDate is not NULL THEN 1 ELSE 0 END as PDDSignedFlag
		,CASE WHEN prod.CreditState like '%Отказан%' THEN 1 ELSE 0 END IsRejected
			       
FROM      [DWH_Stage].[dbo].[BG_RETAIL_PRODUCTION] prod

LEFT JOIN [DWH_Stage].[dbo].[BG_RETAIL_DTS_ELIGIBILITY_REPORT] el on el.CreditCode = prod.CreditNumber

LEFT JOIN (
          select distinct Username UserCode, 
		  ISNULL(p.FirstName, '') + ISNULL(' ' + p.MiddleName, '') + ISNULL( ' ' + p.LastName,'') Username 
		  from [Jetix].[dbo].[tUser] u 
		  left join [Jetix].[dbo].[tPerson] p on p.id=u.id 
		  )as uname on uname.UserCode=prod.EmployeeUID

LEFT JOIN (
          select distinct [Uid] UserCode, 
		  ISNULL(p.FirstName, '') + ISNULL(' ' + p.MiddleName, '') + ISNULL( ' ' + p.LastName,'') Username 
		  from [Jetix].[dbo].[tEmployee] u 
		  left join [Jetix].[dbo].[tPerson] p on p.id=u.id 
		  ) as uidname on uidname.UserCode=prod.EmployeeUID

LEFT JOIN
		  (
		   select   'VRC' as IsVoiceRecording,fkCredit
		   from  [Jetix].[dbo].[tCreditVoiceRecordingContract] vrc with (nolock) 
		   join [Jetix].[dbo].[nResult] r on vrc.fkResult = r.id and r.Enum = 'Success' 		        
		   and cast(DateTimeStarted as date)>= DATEADD(yy,DATEDIFF(yy,0,GETDATE())-4,0)
		   )as IsVRC on fkCredit = prod.id

LEFT JOIN (
           select 1 as IsMobileVisitSent,
		          fkCredit,
				  ROW_NUMBER() over(Partition by fkCredit order by Time desc) row_id
			    --a.Enum Action
		 from jetix.dbo.tCreditActionHistory h
		 left join jetix.dbo.nCreditAction a on a.id = h.fkCreditAction
		 where a.Enum in ('CreditSentFromMobile') 
         and cast(Time as date)>= DATEADD(yy,DATEDIFF(yy,0,GETDATE())-4,0)
		  ) as mobile on mobile.fkCredit=prod.id and row_id = 1

left join (
              select fkCredit,
			  Time as PDDSignedToQuickSignDate,			  
			  ROW_NUMBER() over(Partition by fkCredit order by Time desc) row_nb2 
			  from jetix.dbo.tCreditActionHistory h 
			  where fkCreditAction = 39		   
			  and cast(Time as date)>= DATEADD(yy,DATEDIFF(yy,0,GETDATE())-4,0)
			 )as PDDsigned on PDDsigned.fkCredit=prod.id and row_nb2 = 1	

where prod.StartedOn >= DATEADD(yy,DATEDIFF(yy,0,GETDATE())-4,0)