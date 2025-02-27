--exec [dbo].[TKPI_BG_Product_Insert]
-- =============================================
-- Author:		Deyan Dimitrov
-- Create date: 2010-01-08
-- Description:	Bulk insert on Transformation KPI Product table for Bulgaria
-- =============================================

CREATE PROCEDURE [dbo].[TKPI_BG_Product_Insert]

WITH RECOMPILE
AS
						
BEGIN


declare @Date_DM as date = (select MAX(CAST([Openning_date] as date)) from [BNP-DM-DB].[dm].[dbo].CRM_KPI_Product)

declare @Date_DHW as date = (select MAX(CAST([Openning_date] as date)) from [DWH_Stage].[dbo].[Transformation_KPI_BG_Product])

--------------------------------------------------

IF @Date_DM > ISNULL(@Date_DHW,'1900-01-01')

BEGIN

----truncate table DM server is updated
truncate table [DWH_Stage].[dbo].[Transformation_KPI_BG_Product]

----Insert information into the table 
insert into [DWH_Stage].[dbo].[Transformation_KPI_BG_Product]

select 
	   [ID_product]
      ,[ID_Customer]  
      ,[Openning_date]
      ,[Product_type]
      ,[Product_name]
      ,[Interest_rate]
      ,[Opening_fees]
      ,[Duration]
      ,[Amount]
      ,[ID_Vendor]
      ,[ID_Retailer]
      ,[Flag_crossable]
      ,[Channel]
      ,[Business_line]
      ,[Network]
      ,[Interest_rate_customer]
      ,[Interest_rate_vendor]

 from [BNP-DM-DB].[dm].[dbo].CRM_KPI_Product

 

 ------------------------------------------------------------------ALTER INDEX------------------------------------------------------------------- 

ALTER INDEX ALL ON [DWH_Stage].[dbo].[Transformation_KPI_BG_Product]
REBUILD WITH (FILLFACTOR = 80, SORT_IN_TEMPDB = ON,
              STATISTICS_NORECOMPUTE = ON);

END
 ------------------------------------------------------------------TKPI Product-------------------------------------------------------------------

END