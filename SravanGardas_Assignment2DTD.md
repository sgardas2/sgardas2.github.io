<!-- BrockerCheckReport is the Root -->
<!DOCTYPE brockercheckreport[
<!ELEMENT brockercheckreport (COMPANY+)>
<!--A company is the child of BrokerCheckReport, all the reports will be at this level. -->
<!--Many Companies could be in the BrockerCheckReport and all the companies would have the same schema -->
<!--A Company has three childs FirmProfile, FirmHistory ,FirmOperations -->
<!ELEMENT COMPANY (FirmProfile,FirmHistory,FirmOperations)>
<!--Company has CRD the unique Number to the company and the Name as Mandatory attributes.-->
<!ATTLIST COMPANY  CRD CDATA #REQUIRED>
<!ATTLIST COMPANY  NAME CDATA #REQUIRED>
<!ELEMENT FirmProfile (notes,FirmName,MainOfficeLocation,RegulatedBy,MailingAddress,BusinessTelephoneNumber,DirectOwnersAndExecutiveOfficers+,IndirectOwners*)>
<!--FirmProfile has 8 childs Notes,FirmName,MainOfficeLocation,RegulatedBy,MailingAddress,BusinessTelephoneNumber,DirectOwnersAndExecutiveOfficers,IndirectOwners -->
<!--FirmProfile has one or Many DirectOwners And ExecutiveOfficers and It can have zero or Many Indirect Owners -->
<!ELEMENT notes (#PCDATA)>
<!--Notes is about the data inside FirmProfile -->
<!ELEMENT FirmName (#PCDATA)>
<!--FirmName contains the name of the Firm -->
<!ATTLIST FirmName  CRD CDATA #REQUIRED>
<!ATTLIST FirmName  SEC CDATA #REQUIRED>
<!--Firm Name has two attributes CRD and SEC which are required -->
<!ELEMENT MainOfficeLocation (#PCDATA)>
<!--MainOfficeLocation Gives the address of the Firm -->
<!ELEMENT RegulatedBy (#PCDATA)>
<!--The Office which is regulating the Firm-->
<!ELEMENT MailingAddress (#PCDATA)>
<!--The Mailing address of the Firm-->
<!ELEMENT BusinessTelephoneNumber (#PCDATA)>
<!--The Business Telephone of the Firm-->
<!ELEMENT DirectOwnersAndExecutiveOfficers (LegalNameAndCRD,Position,PositionStartDate,PercentageofOwnership)>
<!--The Element DirectOwnersAndExecutiveOfficers has four childs LegalNameAndCRD of every owner,Position,PositionStartDate,PercentageofOwnership-->
<!ATTLIST DirectOwnersAndExecutiveOfficers OwnerType (Domestic|ForeignEntity|Individual) "Domestic">
<!-- OwnerType attribute of Element DirectOwnersAndExecutiveOfficers  has three controlled which classify the owner as either Domestic/ForeignEntity/Individual.The default here is Domestic -->
<!ATTLIST DirectOwnersAndExecutiveOfficers DirectsPolicies (Yes|No) "No">
<!-- DirectsPolicies attribute of Element DirectOwnersAndExecutiveOfficers  has Yes or No as the controlled list. Default here is No -->
<!ATTLIST DirectOwnersAndExecutiveOfficers PublicCompany (Yes|No) "No">
<!-- PublicCompany attribute of Element DirectOwnersAndExecutiveOfficers  has Yes or No as the controlled list. Default here is No -->
<!ELEMENT LegalNameAndCRD (#PCDATA)>
<!-- LegalNameAndCRD is the child element of DirectOwnersAndExectiveOfficers. It contains the Name and CRD number owner.-->
<!ELEMENT Position (#PCDATA)>
<!-- Position is the child element of DirectOwnersAndExectiveOfficers. It gives the position of the person in the company. -->
<!ELEMENT PositionStartDate (#PCDATA)>
<!-- PositionStartDate gives the Date from when the owner took this position.-->
<!ELEMENT PercentageofOwnership (#PCDATA)>
<!-- PercentageOfOwnership gives the percentage the owner has in this Firm -->
<!ELEMENT IndirectOwners (#PCDATA)>
<!-- IndirectOwners gives the indirect owners of the Firm. This is an optional element.-->
<!ELEMENT FirmHistory (#PCDATA)>
<!-- Firm History is the second child of Company. This gives the history of Megers and Acquistions-->
<!ELEMENT FirmOperations (Registrations,TypesofBusiness,ClearingArrangements,IntroducingArrangements,IndustryArrangements,OrganizationAffiliates)>
<!-- Firm Operations is the Third and the last child of Company. This gives the operations information about the company-->
<!-- Firm Operations has 6 childs. They specify about Registration,TypesOfBusiness the firm does, Clearing Arragements, IntroductingArragements,IndustryArragements,OrganizationAffiliates of the firm-->
<!ELEMENT Registrations (Notes,FederalRegulator,SECRegistrationQuestions,Self-RegulatoryOrganization,U.S.StatesandTerritories)>
<!-- Registration Element has five child elements. They gives about the registration entity ,registrations which the firm possess and the states it can operate.-->
<!ELEMENT Notes (#PCDATA)>
<!-- The element Notes gives brief description of Registration -->
<!ELEMENT FederalRegulator EMPTY>
<!-- The element FederalRegulator  is an empty element. It has two attributes Name of the approval regualatory. The status of the approaval and Date from which this approval was effective -->
<!ATTLIST FederalRegulator Name  CDATA #IMPLIED>
<!ATTLIST FederalRegulator Status CDATA #IMPLIED>
<!ATTLIST FederalRegulator DateEffective  CDATA #IMPLIED>
<!-- The element SECRegistrationQuestions is an empty element with four attributes which provides various details about registration the firm -->
<!ELEMENT SECRegistrationQuestions EMPTY>
<!-- The broker-dealer attribute specifies if the firm is a Broker Dealer or not. It has two values Yes/No with No being default -->
<!ATTLIST SECRegistrationQuestions broker-dealer (Yes|No) "No">
<!-- The GovernmentSecuritiesBrokerattribute specifies if the firm is a Government Securities Broker or not. It has two values Yes/No with No being default -->
<!ATTLIST SECRegistrationQuestions GovernmentSecuritiesBroker (Yes|No) "No">
<!-- The OnlyGovernmentSecuritiesBroker attribute specifies if the firm is a OnlyGovernment Securities Broker or not. It has two values Yes/No with No being default -->
<!ATTLIST SECRegistrationQuestions OnlyGovernmentSecuritiesBroker (Yes|No) "No">
<!-- The CeasedGovernamentSecurityActivties attribute specifies if the firm cesaedGovernament Security activties or not. It has two values Yes/No with No being default -->
<!ATTLIST SECRegistrationQuestions CeasedGovernamentSecurityActivties (Yes|No) "No">
<!ELEMENT Self-RegulatoryOrganization EMPTY >
<!-- The element Self-RegulatoryOrganization is an empty element with three attributes which provides various details about  the self regulatory organization of the firm -->
<!ATTLIST Self-RegulatoryOrganization Name  CDATA #IMPLIED>
<!-- The Self-RegulatoryOrganization Name, Status and Date Effective attributes, specify the details of the self regulatory organization -->
<!ATTLIST Self-RegulatoryOrganization Status CDATA #IMPLIED>
<!ATTLIST Self-RegulatoryOrganization DateEffective  CDATA #IMPLIED>
<!-- The element U.S.StatesandTerritories specfies all the states in which the firm can operate. It has state as child element, which can have one or many states. -->
<!ELEMENT U.S.StatesandTerritories (state+)>
<!-- The element state is empty. It has three attributes Name,Status and DateEffective of the state in which it operates -->
<!ELEMENT state EMPTY>
<!ATTLIST state Name  CDATA #IMPLIED>
<!ATTLIST state Status CDATA #IMPLIED>
<!ATTLIST state DateEffective  CDATA #IMPLIED>
<!-- The element TypesofBusiness specifies all the business the firm does. It has two child elements, with Typeofbusiness being a child which can have one or more occurences-->
<!ELEMENT TypesofBusiness (TBNotes,TypeofBusiness+)>
<!ELEMENT TBNotes (#PCDATA)>
<!-- TBNotes gives notes or description about TypesOfBusiness. Also it gives a summary-->
<!ELEMENT TypeofBusiness (#PCDATA)>
<!-- TypeofBusiness contains the business which can firm currently does.-->
<!ELEMENT ClearingArrangements (CANotes)>
<!-- Clearning Arrangements contains the clearning arragements of the Firm. The Child element CANotes gives notes or description about Clearing Arrangements-->
<!ELEMENT CANotes (#PCDATA)>
<!ELEMENT IntroducingArrangements (IANotes,Name,BusinessAddress,CRD,EffectiveDate,Description)>
<!-- IntroducingArrangements contains the Introducing Arrangementsof the Firm. It has six childs which provide additional details about it -->
<!ELEMENT IANotes (#PCDATA)>
<!-- IANotes gives notes or description about IntroducingArrangements. Also it gives a summary-->
<!ELEMENT Name (#PCDATA)>
<!-- Gives Name of the company introduced-->
<!ELEMENT BusinessAddress (#PCDATA)>
<!-- Gives BusinessAddress of the company introduced-->
<!ELEMENT CRD (#PCDATA)>
<!-- Gives CRD of the company introduced-->
<!ELEMENT EffectiveDate (#PCDATA)>
<!-- Gives EffectiveDate  from which the company was introduced-->
<!ELEMENT Description (#PCDATA)>
<!-- Gives Descrition of the relationship to the  company which was introduced-->
<!ELEMENT IndustryArrangements (INANotes)>
<!-- This element Gives IndustryArrangements  of the Firm-->
<!ELEMENT INANotes (#PCDATA)>
<!-- This element NANotes gives summary of the IndustryArrangements  of the Firm-->
<!ELEMENT OrganizationAffiliates (OANotes,Firm+,AdditionalNotes)>
<!-- This element OrganizationAffiliates gives OrganizationAffiliates of the Firm. It has three childs, with Firm being a child with one or many occurences-->
<!ELEMENT OANotes (#PCDATA)>
<!-- This element OANotes gives summary of the Organization affiliates of the Firm-->
<!ELEMENT Firm (Control,FirmAddress,Country,FirmDescription)>
<!-- This element Firm has four childs which give the details about the control the firm has, the address of the firm, the country in which the firm is location and descriptio of the firm-->
<!ATTLIST Firm ForeignEntity (Yes|No) "No">
<!-- The attribute ForeignEntity  specify if the firm is ForeignEntity or not with Yes/No values. The default being No -->
<!ATTLIST Firm SecuritiesActivities (Yes|No) "No">
<!-- The attribute SecuritiesActivities  specify if the firm does securityActivites or not with Yes/No values. The default being No -->
<!ATTLIST Firm InvestmentAdvisoryActivities (Yes|No) "No">
<!-- The attribute InvestmentAdvisoryActivities  specify if the firm does advisory activites or not with Yes/No values. The default being No -->
<!ATTLIST Firm Name CDATA #IMPLIED>
<!-- The attribute Firm Name specify the name of the Firm. -->
<!ATTLIST Firm EffectiveDate CDATA #IMPLIED>
<!-- The attribute EffectiveDate specify effective date of the firm -->
<!ELEMENT Control (#PCDATA)>
<!-- The Element control specify the controling details of the firm -->
<!ELEMENT FirmAddress (#PCDATA)>
<!-- The Element FirmAddress gives the address of the firm -->
<!ELEMENT Country (#PCDATA)>
<!-- The Element country give the country where the firm is located-->
<!ELEMENT FirmDescription (#PCDATA)>
<!-- The Element FirmDescription give descrition of the firm -->
<!ELEMENT  AdditionalNotes (#PCDATA)>
<!-- The Element AdditionalNotes, provides additional notes on the Firm -->
]>
