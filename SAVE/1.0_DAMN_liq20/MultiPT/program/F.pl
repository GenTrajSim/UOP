sub whether_in {
	my ($panding,$OP_t) = @_;
	foreach my $u(@$OP_t){
		#print "$u ";
		if ($u == $panding) { return 1; }
	}
	return 0;
}

open (INOP,"<OP.txt");
while (<INOP>) {push @in, [split];}
my @total_index = ();
my $num_index = 0;
foreach my $b(@in) {
	my $lin_n = $b -> [$num_index];
	while (($lin_n)||($num_index==0)) {
		$total_index[$num_index] = ($lin_n);  ########tip6p (6) and protein (1031)
		$num_index ++ ;
		$lin_n = $b -> [$num_index];
		#print "$lin_n ";
	}
	#print "\n";
	
}
close INOP;

open (INCJ,"<cj1.txt");
open OUT, "+> F.txt";
while (<INCJ>) {
	push @incj,[split];
}
$ii = 0;
foreach my $a(@incj) {
	$ii++;
	my $centro = $a -> [0];
	#print "$centro\n";
	if ( &whether_in($centro,\@total_index) ) {
		my $num_neb = $a -> [1];
		my $neb_index1 = $a ->[2];
		my $neb_index2 = $a ->[3];
		my $neb_index3 = $a ->[4];
		my $neb_index4 = $a ->[5];
		my $neb_index5 = $a ->[6];
		my $neb_index6 = $a ->[7];
		#print  "$centro $total_index $num_neb $neb_index1 $neb_index2 $neb_index3 $neb_index4 $neb_index5 $neb_index6\n";
		print OUT "$centro $total_index $num_neb $neb_index1 $neb_index2 $neb_index3 $neb_index4 $neb_index5 $neb_index6\n";
	}
	
}
@incj=();
close INCJ;
close OUT;
