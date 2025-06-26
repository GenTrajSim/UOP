for (my $i = 1; $i<=100; $i++) {
	my $name1 = 'dump.'.($i*1000).'.xyz';
	my $name2 = '4.300.0.'.($i*1000).'.Urea.xyz';
	print "mv $name1 $name2\n";
	system("mv $name1 $name2");
}
