<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  version="1.0"
  xmlns:alto="http://www.loc.gov/standards/alto/ns-v3#"
>
  <xsl:output method="text"/>
  <xsl:strip-space elements="*"/>

  <xsl:template match="alto:String">
    <xsl:if test="position() > 1">
      <xsl:text> </xsl:text>
    </xsl:if>
    <xsl:value-of select="@CONTENT" />
  </xsl:template>

  <xsl:template match="alto:TextLine">
    <xsl:apply-templates select=".//alto:String"/>
    <!-- always end line with line feed -->
    <xsl:text>&#10;</xsl:text>
  </xsl:template>

  <xsl:template match="alto:TextBlock">
    <xsl:if test="position() > 1">
      <!-- blank line between blocks -->
      <xsl:text>&#10;</xsl:text>
    </xsl:if>
    <xsl:apply-templates select=".//alto:TextLine[.//alto:String]"/>
  </xsl:template>

  <xsl:template match="alto:Page">
    <xsl:if test="position() > 1">
      <!-- two blank lines between pages -->
      <xsl:text>&#10;&#10;</xsl:text>
    </xsl:if>
    <xsl:apply-templates select=".//alto:TextBlock[.//alto:String]"/>
  </xsl:template>

  <xsl:template match="/">
    <xsl:apply-templates select=".//alto:Page"/>
    <xsl:text>&#10;</xsl:text>
  </xsl:template>
</xsl:stylesheet>
