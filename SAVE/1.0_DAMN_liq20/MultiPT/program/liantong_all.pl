sub whether_in{
	my ($panding_i,$total_i) = @_;
	foreach my $uu(@$total_i) {
		if ($uu == $panding_i) {return 0;}
	}
	return 1;
}

sub LT_0 {
	my $first_cent = $_[0] ;
	my $first_type = $_[1] ;
	my $zushu = $_[2] ;
	my $Pd_0 = 1; 
	for ( my $lin_i = 1 ; $lin_i <= $first_type ; $lin_i ++ ) {
		$Pd_0 = $Pd_0 * $cent_type[$d_1[$first_cent][$lin_i]] ;
	}
	if ( $Pd_0 == 0 ) {
		for ( my $lin_i = 1 ; $lin_i <= $first_type ; $lin_i ++ ) {
			if ( $cent_type[$d_1[$first_cent][$lin_i]] == 0 ) {
				$cent_type[$d_1[$first_cent][$lin_i]] = $zushu ;
				&LT_0( $d_1[$first_cent][$lin_i] , $type[$d_1[$first_cent][$lin_i]] , $cent_type[$d_1[$first_cent][$lin_i]] ) ;
			}
		}
	}
}

open (FI,"<F.txt") ;
open FP,'+> Uni-OP.txt';

while (<FI>){
	push @fi ,[split] ;
}

$i = 0 ;

foreach my $a(@fi) {
	$i ++ ;
	$cent_0[$i] = $a -> [0] ;
	$type[$cent_0[$i]] = $a -> [1] ;
	$d_1[$cent_0[$i]][1] = $a -> [2] ;
	$d_1[$cent_0[$i]][2] = $a -> [3] ;
	$d_1[$cent_0[$i]][3] = $a -> [4] ;
	$d_1[$cent_0[$i]][4] = $a -> [5] ;
	$d_1[$cent_0[$i]][5] = $a -> [6] ;
	$d_1[$cent_0[$i]][6] = $a -> [7] ;
	$d_1[$cent_0[$i]][7] = $a -> [8] ;
	$d_1[$cent_0[$i]][8] = $a -> [9] ;
	$d_1[$cent_0[$i]][9] = $a -> [10] ;
	$d_1[$cent_0[$i]][10] = $a -> [11] ;
	$d_1[$cent_0[$i]][11] = $a -> [12] ;
	$d_1[$cent_0[$i]][12] = $a -> [13] ;
	$d_1[$cent_0[$i]][13] = $a -> [14] ;
	$d_1[$cent_0[$i]][14] = $a -> [15] ;
	$d_1[$cent_0[$i]][15] = $a -> [16] ;
	$cent_type[$cent_0[$i]] = 0 ;
}

$zhushu = 1 ;
$jishu_zhu[$zhushu] = 0 ;

for ( $j = 1 ; $j <= $i ; $j ++ ) {
	if ( ($cent_type[$cent_0[$j]] == 0) ) {
		&LT_0( $cent_0[$j] , $type[$cent_0[$j]] , $zhushu );
		$zhushu ++
	}
}

for (my $fff = 1 ; $fff <= $zhushu ; $fff ++ ) {
	my @lin_index = ();
	$jishu_zhu[$fff] = 0;
	for ( $j = 1 ; $j <= $i ; $j ++ ) {
		if ( ($cent_type[$cent_0[$j]] == $fff) && (&whether_in($cent_0[$j],\@lin_index)) ) {
			$lin_index[$jishu_zhu[$fff]] = $cent_0[$j];
			$jishu_zhu[$fff] ++ ;
		}
	}
}
$max = $jishu_zhu[$zhushu] ;
$max_i = $zhushu ;
$max_i_f1 = $zhushu ;
$max_f1 = $jishu_zhu[$zhushu] ;

print "$zhushu $max_f1\n";
print "$jishu_zhu[$zhushu-3]\n";

my @px = ();
$px[0] =10000;
my @pxi = ();
my $pxn = 0;
for (my $px1 = 1; $px1< $zhushu ; $px1++) {
	my $max_up = $px1 -1;
	my $min_do = $px1;
	for (my $px2 =1; $px2< $zhushu ; $px2++) {
		if ( ($jishu_zhu[$px2] > $px[$min_do]) && ($jishu_zhu[$px2]<=$px[$max_up]) ) {
			if (&whether_in($px2,\@pxi)) {
				if ($jishu_zhu[$px2]) {
					#print "($px2 -> $jishu_zhu[$px2])";
				}
				#print "$px1:($px[$min_do])=>";
				#print "Y($jishu_zhu[$px2])";
				$px[$min_do] = $jishu_zhu[$px2];
				$pxi[$min_do] = $px2;
			}
		}
	}
	#print "\n";
	if ($jishu_zhu[$pxi[$min_do]]) {
		print "### $min_do :  $px[$min_do] $pxi[$min_do] $jishu_zhu[$pxi[$min_do]]\n";
	}
	#
}

for (my $fff = 1 ; $fff < $zhushu ; $fff ++ ) {
	if ( $jishu_zhu[$fff] > $max ) {
		$max = $jishu_zhu[$fff] ;
		$max_i = $fff ;
	}
}
#print FP "$max | ";
#for ( $j = 1 ; $j <= $i ; $j ++ ) {
#	if ( $cent_type[$cent_0[$j]] == $max_i ) {
#		print FP " $cent_0[$j] ";
#	}
#}
#print FP "\n";

for (my $px1 = 1; $px1< $zhushu ; $px1++) {
	if ($jishu_zhu[$pxi[$px1]]) {
		print FP "$px[$px1] | ";
		for ( $j = 1 ; $j <= $i ; $j ++ ) {
			if ( $cent_type[$cent_0[$j]] == $pxi[$px1] ) {
				print FP " $cent_0[$j] ";
			}
		}
		print FP "\n";
	}
}

exit 1;
