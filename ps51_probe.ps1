# Decisive probe: does a SCALAR .Count answer $null or 1, per type, on this host?
$ErrorActionPreference = "Continue"
function Show($label, $v) {
    $t = if ($null -eq $v) { "<null>" } else { $v.GetType().FullName }
    $c = $v.Count
    $cs = if ($null -eq $c) { '$null' } else { "$c" }
    $isArr = $v -is [array]
    "{0,-46} type={1,-58} isArray={2,-5} .Count={3}" -f $label, $t, $isArr, $cs
}
"HOST PSVersion = $($PSVersionTable.PSVersion)  Edition=$($PSVersionTable.PSEdition)"
"=" * 150
# --- plain scalars ---
$s = 'a';            Show "string scalar" $s
$i = 4;              Show "int scalar" $i
$pso = [pscustomobject]@{ a = 1 };  Show "[pscustomobject] scalar" $pso
$ht  = @{ a = 1 };   Show "hashtable" $ht
"=" * 150
# --- the unroll, on a plain array ---
$one = @('only')
$unwrapped = if ($one.Count -gt 0) { $one } else { @() }
$wrapped   = @(if ($one.Count -gt 0) { $one } else { @() })
Show "if-expr, 1-elem [string[]] -> unwrapped" $unwrapped
Show "if-expr, 1-elem [string[]] -> @() wrapped" $wrapped
"=" * 150
# --- THE ACTUAL SHAPE FROM studio/setup.ps1, on a real single-instance CIM class ---
# Win32_OperatingSystem is guaranteed to return exactly one instance, which is the
# single-AMD-GPU case from #8335 without needing an AMD GPU.
$amdGpus = @(Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue)
$healthyGpus = @($amdGpus | Where-Object { $true })
"amdGpus.Count=$($amdGpus.Count)  healthyGpus.Count=$($healthyGpus.Count)"
$wmiOld = if ($healthyGpus.Count -gt 0) { $healthyGpus } else { $amdGpus }
$wmiNew = @(if ($healthyGpus.Count -gt 0) { $healthyGpus } else { $amdGpus })
Show "OLD  `$wmiGpus = if (...)   [CimInstance]" $wmiOld
Show "NEW  `$wmiGpus = @(if (...)) [CimInstance]" $wmiNew
"OLD branch taken? (`$wmiGpus.Count -gt 0) = $($wmiOld.Count -gt 0)"
"NEW branch taken? (`$wmiGpus.Count -gt 0) = $($wmiNew.Count -gt 0)"
"=" * 150
# --- same, on the REAL class setup.ps1 uses, whatever adapters this runner has ---
$vc = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue)
"Win32_VideoController count = $($vc.Count) :: $(($vc | ForEach-Object { $_.Name }) -join ' | ')"
if ($vc.Count -ge 1) {
    $justOne = @($vc[0])
    $vcOld = if ($justOne.Count -gt 0) { $justOne } else { @() }
    Show "OLD, forced single Win32_VideoController" $vcOld
    "OLD branch taken? = $($vcOld.Count -gt 0)"
}
"=" * 150
# --- deserialized / PSObject-wrapped variants, in case remoting-style wrapping matters ---
$deser = [System.Management.Automation.PSSerializer]::Deserialize([System.Management.Automation.PSSerializer]::Serialize(@($amdGpus)))
$d1 = @($deser)
$dOld = if ($d1.Count -gt 0) { $d1 } else { @() }
Show "OLD, deserialized CimInstance" $dOld
"=" * 150
"DONE"
